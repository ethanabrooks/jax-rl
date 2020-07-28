import itertools
from dataclasses import dataclass
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from flax import optim
from haiku import PRNGSequence
from jax import random

from flax_models import (
    build_gaussian_policy_model,
    build_double_critic_model,
    build_constant_model,
)
from saving import save_model, load_model
from utils import double_mse, apply_model, copy_params

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

        actor_input_dim = [((1, *state_shape), jnp.float32)]

        self.actor = build_gaussian_policy_model(
            actor_input_dim, action_dim, max_action, next(self.rng)
        )

        init_rng = next(self.rng)

        def actor(obs, key=None):
            return GaussianPolicy(action_dim=action_dim, max_action=max_action)(
                obs, key
            )

        def critic(obs, action):
            return DoubleCritic()(obs, action)

        def log_alpha(_=None):
            return Constant()(initial_log_alpha)

        self.critic_input_dim = [
            ((1, *state_shape), jnp.float32),
            ((1, action_dim), jnp.float32),
        ]
        self.critic = build_double_critic_model(self.critic_input_dim, init_rng)

        def transform(f) -> hk.Transformed:
            return hk.without_apply_rng(hk.transform(f, apply_rng=True))

        self.net = Nets(
            actor=transform(actor),
            critic=transform(critic),
            target_critic=transform(critic),
            log_alpha=transform(log_alpha),
        )
        self.entropy_tune = entropy_tune
        self.log_alpha = build_constant_model(-3.5, next(self.rng))
        self.target_entropy = -action_dim

        self.adam = Optimizers(
            actor=optim.Adam(learning_rate=lr),
            critic=optim.Adam(learning_rate=lr),
            log_alpha=optim.Adam(learning_rate=lr),
        )
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
        self.critic_target = build_double_critic_model(
            self.critic_input_dim, next(self.rng)
        )
        self.flax_optimizer = Optimizers(
            actor=(self.adam.actor.create(self.actor)),
            critic=(self.adam.critic.create(self.critic)),
            log_alpha=(self.adam.log_alpha.create(self.log_alpha)),
        )
        self.flax_optimizer.actor = jax.device_put(self.flax_optimizer.actor)
        self.flax_optimizer.critic = jax.device_put(self.flax_optimizer.critic)
        self.flax_optimizer.log_alpha = jax.device_put(self.flax_optimizer.log_alpha)
        return vars(params), vars(opt_params)

    def get_td_target(
        self,
        critic_target,
        rng: PRNGKey,
        params: Params,
        next_obs: jnp.ndarray,
        reward: jnp.ndarray,
        not_done: jnp.ndarray,
    ):
        # next_action, next_log_p = self.net.actor.apply(params.actor, next_obs, rng)
        next_action, next_log_p = self.flax_optimizer.actor.target(
            next_obs, sample=True, key=rng
        )

        target_Q1, target_Q2 = critic_target(next_obs, next_action)
        target_Q = (
            jnp.minimum(target_Q1, target_Q2)
            - jnp.exp(self.net.log_alpha.apply(params.log_alpha)) * next_log_p
        )
        target_Q = reward + not_done * self.discount * target_Q

        return target_Q

    # noinspection PyPep8Naming
    @staticmethod
    def critic_loss(
        params: jnp.ndarray,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        target_Q: jnp.ndarray,
    ):
        current_Q1, current_Q2 = params(obs, action)
        # current_Q1, current_Q2 = self.net.critic.apply(params, obs, action)
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

    # @functools.partial(jax.jit, static_argnums=0)
    def update_critic(
        self, params: dict, opt_params: dict, obs, action, **kwargs,
    ):
        params = Params(**params)
        opt_params = OptParams(**opt_params)

        target_Q = jax.lax.stop_gradient(
            self.get_td_target(
                rng=next(self.rng),
                **kwargs,
                critic_target=self.critic_target,
                params=params,
            )
        )

        self.flax_optimizer.critic = self.critic_step(
            critic=self.flax_optimizer.critic,
            state=obs,
            action=action,
            target_Q=target_Q,
            opt_params=opt_params.critic,
        )

        # grad = jax.grad(self.critic_loss)(
        #     params.critic, obs=obs, action=action, target_Q=target_Q
        # )
        # params.critic, opt_params.critic = self.apply_updates(
        #     grad=grad,
        #     optimizer=self.optimizer.critic,
        #     _params=params.critic,
        #     _opt_params=opt_params.critic,
        # )
        #
        return vars(params), vars(opt_params)

    @functools.partial(jax.jit, static_argnums=0)
    def actor_step(self, rng, actor, critic, state, log_alpha, opt_params):
        def loss_fn(actor):
            actor_action, log_p = self.net.actor.apply(actor, state, key=rng)
            q1, q2 = critic(state, actor_action)
            min_q = jnp.minimum(q1, q2)
            partial_loss_fn = jax.vmap(
                partial(
                    actor_loss_fn,
                    jax.lax.stop_gradient(self.net.log_alpha.apply(log_alpha)),
                )
            )
            actor_loss = partial_loss_fn(log_p, min_q)
            return jnp.mean(actor_loss), log_p

        grad, log_p = jax.grad(loss_fn, has_aux=True)(actor)
        # return optimizer.apply_gradient(grad), log_p
        updates, opt_params = self.optimizer.log_alpha.update(grad, opt_params)
        return optix.apply_updates(actor, updates), opt_params, log_p

    @functools.partial(jax.jit, static_argnums=0)
    def critic_step(self, critic, opt_params, state, action, target_Q):
        def loss_fn(critic):
            current_Q1, current_Q2 = critic(state, action)
            critic_loss = double_mse(current_Q1, current_Q2, target_Q)
            return jnp.mean(critic_loss)

        grad = jax.grad(loss_fn)(critic.target)
        return critic.apply_gradient(grad)

    @functools.partial(jax.jit, static_argnums=0)
    def alpha_step(self, params, opt_params, log_p, target_entropy):
        log_p = jax.lax.stop_gradient(log_p)

        def loss_fn(params):
            partial_loss_fn = jax.vmap(
                partial(alpha_loss_fn, self.net.log_alpha.apply(params), target_entropy)
            )
            return jnp.mean(partial_loss_fn(log_p))

        grad = jax.grad(loss_fn)(params)
        updates, opt_params = self.optimizer.log_alpha.update(grad, opt_params)
        return optix.apply_updates(params, updates), opt_params

    def update_actor(self, params, opt_params, state):
        params = Params(**params)
        opt_params = OptParams(**opt_params)

        params.actor, opt_params.actor, log_p = self.actor_step(
            rng=next(self.rng),
            actor=params.actor,
            critic=self.flax_optimizer.critic.target,
            state=state,
            log_alpha=params.log_alpha,
            opt_params=opt_params.actor,
        )

        if self.entropy_tune:
            params.log_alpha, opt_params.log_alpha = self.alpha_step(
                params=params.log_alpha,
                opt_params=opt_params.log_alpha,
                log_p=log_p,
                target_entropy=self.target_entropy,
            )

        self.critic_target = self.critic_target.replace(
            params=copy_params(
                self.flax_optimizer.critic.target.params,
                self.critic_target.params,
                self.tau,
            )
        )

        return vars(params), vars(opt_params)

    def select_action(self, params, obs):
        # mu, _ = apply_model(self.flax_optimizer.actor.target, state)
        mu, log_sig = self.net.actor.apply(params, obs)
        return mu.flatten()

    def sample_action(self, rng, params, obs):
        # mu, log_sig = apply_model(self.flax_optimizer.actor.target, state)
        mu, log_sig = self.net.actor.apply(params, obs)
        return mu + random.normal(rng, mu.shape) * jnp.exp(log_sig)

    def train(self, replay_buffer, batch_size=100, load_path=None):
        if self.iterator is None:
            self.iterator = self.generator(load_path=load_path)
            next(self.iterator)

        data = replay_buffer.sample(next(self.rng), batch_size)
        return self.iterator.send(data)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, params, obs, rng=None):
        mu, log_sig = self.net.actor.apply(params, obs)
        if rng is not None:  # TODO: this is gross
            mu += random.normal(rng, mu.shape) * jnp.exp(log_sig)
        return mu
