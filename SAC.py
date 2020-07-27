import functools
from dataclasses import dataclass
from functools import partial

import haiku as hk
import jax
import jax.experimental.optix as optix
import jax.numpy as jnp
from haiku import PRNGSequence
from haiku._src.typing import PRNGKey
from jax import random

from models import (
    DoubleCritic,
    GaussianPolicy,
    Constant,
)
from utils import double_mse


@dataclass
class Nets:
    T = hk.Transformed
    actor: T
    critic: T
    target_critic: T
    log_alpha: T


@dataclass
class Optimizers:
    T = optix.GradientTransformation
    actor: T
    critic: T
    log_alpha: T


@dataclass
class Params:
    T = jnp.array
    actor: T
    critic: T
    target_critic: T
    log_alpha: T


@dataclass
class OptParams:
    T = jnp.array
    actor: T
    critic: T
    log_alpha: T


def actor_loss_fn(log_alpha, log_p, min_q):
    return (jnp.exp(log_alpha) * log_p - min_q).mean()


def alpha_loss_fn(log_alpha, target_entropy, log_p):
    return (log_alpha * (-log_p - target_entropy)).mean()


class SAC:
    def __init__(
        self,
        state_shape,
        action_dim,
        max_action,
        save_freq,
        discount=0.99,
        tau=0.005,
        actor_freq=2,
        lr=3e-4,
        entropy_tune=True,
        seed=0,
        initial_log_alpha=-3.5,
    ):

        self.rng = PRNGSequence(seed)

        def actor(obs, key=None):
            return GaussianPolicy(action_dim=action_dim, max_action=max_action)(
                obs, key
            )

        def critic(obs, action):
            return DoubleCritic()(obs, action)

        def log_alpha(_=None):
            return Constant()(initial_log_alpha)

        def transform(f) -> hk.Transformed:
            return hk.without_apply_rng(hk.transform(f, apply_rng=True))

        self.net = Nets(
            actor=transform(actor),
            critic=transform(critic),
            target_critic=transform(critic),
            log_alpha=transform(log_alpha),
        )
        self.entropy_tune = entropy_tune
        self.target_entropy = -action_dim

        self.optimizer = Optimizers(
            actor=optix.adam(learning_rate=lr),
            critic=optix.adam(learning_rate=lr),
            log_alpha=optix.adam(learning_rate=lr),
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.actor_freq = actor_freq
        self.save_freq = save_freq

        self.total_it = 0

    def init(self, obs, action):
        key = next(self.rng)
        critic_params = self.net.critic.init(key, obs, action)
        params = Params(
            actor=self.net.actor.init(key, obs, key=None),
            critic=critic_params,
            target_critic=critic_params,
            log_alpha=self.net.log_alpha.init(key),
        )
        opt_params = OptParams(
            actor=self.optimizer.actor.init(params.actor),
            critic=self.optimizer.critic.init(params.critic),
            log_alpha=self.optimizer.log_alpha.init(params.log_alpha),
        )
        return vars(params), vars(opt_params)

    def get_td_target(
        self,
        rng: PRNGKey,
        params: Params,
        next_obs: jnp.ndarray,
        reward: jnp.ndarray,
        not_done: jnp.ndarray,
    ):
        next_action, next_log_p = self.net.actor.apply(params.actor, next_obs, rng)

        target_Q1, target_Q2 = self.net.target_critic.apply(
            params.critic, next_obs, next_action
        )
        target_Q = (
            jnp.minimum(target_Q1, target_Q2)
            - jnp.exp(self.net.log_alpha.apply(params.log_alpha)) * next_log_p
        )
        target_Q = reward + not_done * self.discount * target_Q

        return target_Q

    # noinspection PyPep8Naming
    def critic_loss(
        self,
        params: jnp.ndarray,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        target_Q: jnp.ndarray,
    ):
        current_Q1, current_Q2 = self.net.critic.apply(params, obs, action)
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)

    def actor_loss(
        self,
        actor: jnp.ndarray,
        critic: jnp.ndarray,
        log_alpha: jnp.ndarray,
        obs: jnp.ndarray,
        key: PRNGKey,
    ):
        actor_action, log_p = self.net.actor.apply(actor, obs, key=key)
        q1, q2 = self.net.critic.apply(critic, obs, actor_action)
        min_q = jnp.minimum(q1, q2)
        partial_loss_fn = jax.vmap(
            partial(
                actor_loss_fn,
                jax.lax.stop_gradient(self.net.log_alpha.apply(log_alpha)),
            )
        )
        actor_loss = partial_loss_fn(log_p, min_q)
        return jnp.mean(actor_loss), log_p

    def alpha_loss(
        self, params: jnp.ndarray, log_p: jnp.ndarray,
    ):
        partial_loss_fn = jax.vmap(
            partial(
                alpha_loss_fn, self.net.log_alpha.apply(params), self.target_entropy
            )
        )
        return jnp.mean(partial_loss_fn(log_p))

    @staticmethod
    def apply_updates(
        grad, _params, _opt_params, optimizer: optix.GradientTransformation,
    ):
        updates, _opt_params = optimizer.update(grad, _opt_params)
        return (optix.apply_updates(_params, updates), _opt_params)

    @functools.partial(jax.jit, static_argnums=0)
    def update_critic(self, params: dict, opt_params: dict, obs, action, **kwargs):
        params = Params(**params)
        opt_params = OptParams(**opt_params)

        target_Q = jax.lax.stop_gradient(
            self.get_td_target(rng=next(self.rng), params=params, **kwargs,)
        )

        params.critic, opt_params.critic = self.apply_updates(
            grad=jax.grad(self.critic_loss)(
                params.critic, obs=obs, action=action, target_Q=target_Q
            ),
            optimizer=self.optimizer.critic,
            _params=params.critic,
            _opt_params=opt_params.critic,
        )

        return vars(params), vars(opt_params)

    @functools.partial(jax.jit, static_argnums=0)
    def update_actor(self, params: dict, opt_params: dict, obs: jnp.ndarray):
        params = Params(**params)
        opt_params = OptParams(**opt_params)
        grad, log_p = jax.grad(self.actor_loss, has_aux=True)(
            params.actor,
            critic=params.critic,
            log_alpha=params.log_alpha,
            obs=obs,
            key=next(self.rng),
        )
        params.actor, opt_params.actor = self.apply_updates(
            grad=grad,
            optimizer=self.optimizer.critic,
            _params=params.actor,
            _opt_params=opt_params.actor,
        )

        if self.entropy_tune:
            params.log_alpha, opt_params.log_alpha = self.apply_updates(
                grad=jax.grad(self.alpha_loss)(params.log_alpha, log_p=log_p),
                optimizer=self.optimizer.log_alpha,
                _params=params.log_alpha,
                _opt_params=opt_params.log_alpha,
            )

        params.target_critic = jax.tree_multimap(
            lambda p1, p2: self.tau * p1 + (1 - self.tau) * p2,
            params.target_critic,
            params.critic,
        )

        return vars(params), vars(opt_params)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, params, obs, rng=None):
        mu, log_sig = self.net.actor.apply(params, obs)
        if rng is not None:  # TODO: this is gross
            mu += random.normal(rng, mu.shape) * jnp.exp(log_sig)
        return mu
