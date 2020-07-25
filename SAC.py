import functools
import itertools
from dataclasses import dataclass
from functools import partial
from typing import Union, Callable
import jax.experimental.optix as optix
import jax
import jax.numpy as jnp
from flax import optim
from haiku import PRNGSequence
from haiku._src.typing import PRNGKey
from jax import random
import haiku as hk

from models import (
    build_gaussian_policy_model,
    build_double_critic_model,
    build_constant_model,
    DoubleCritic,
    GaussianPolicy,
    Constant,
)
from saving import save_model, load_model
from utils import double_mse, apply_model, copy_params


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


@jax.jit
def critic_step(optimizer, state, action, target_Q):
    def loss_fn(critic):
        current_Q1, current_Q2 = critic(state, action)
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


@jax.jit
def actor_step(rng, optimizer, critic, state, log_alpha):
    critic, log_alpha = critic.target, log_alpha.target

    def loss_fn(actor):
        actor_action, log_p = actor(state, sample=True, key=rng)
        q1, q2 = critic(state, actor_action)
        min_q = jnp.minimum(q1, q2)
        partial_loss_fn = jax.vmap(
            partial(actor_loss_fn, jax.lax.stop_gradient(log_alpha()))
        )
        actor_loss = partial_loss_fn(log_p, min_q)
        return jnp.mean(actor_loss), log_p

    grad, log_p = jax.grad(loss_fn, has_aux=True)(optimizer.target)
    return optimizer.apply_gradient(grad), log_p


@jax.jit
def alpha_step(optimizer, log_p, target_entropy):
    log_p = jax.lax.stop_gradient(log_p)

    def loss_fn(log_alpha):
        partial_loss_fn = jax.vmap(partial(alpha_loss_fn, log_alpha(), target_entropy))
        return jnp.mean(partial_loss_fn(log_p))

    grad = jax.grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad)


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

        # actor_input_dim = [((1, *state_shape), jnp.float32)]

        # self.actor = build_gaussian_policy_model(
        #     actor_input_dim, action_dim, max_action, next(self.rng)
        # )
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
        # self.critic_input_dim = [
        #     ((1, *state_shape), jnp.float32),
        #     ((1, action_dim), jnp.float32),
        # ]
        # self.critic = build_double_critic_model(self.critic_input_dim, init_rng)
        self.entropy_tune = entropy_tune
        # self.log_alpha = build_constant_model(-3.5, next(self.rng))
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

    # @jax.jit
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
        target_critic: jnp.ndarray,
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
        *args,
        loss,
        _params,
        _opt_params,
        optimizer: optix.GradientTransformation,
        has_aux=False
    ):
        grad = jax.grad(loss, has_aux=has_aux)(*args)
        grad, aux = grad if has_aux else (grad, None)
        updates, _opt_params = optimizer.update(grad, _opt_params)
        return (optix.apply_updates(_params, updates), _opt_params, aux)

    @functools.partial(jax.jit, static_argnums=0)
    def update_critic(self, params: dict, opt_params: dict, obs, action, **kwargs):
        params = Params(**params)
        opt_params = OptParams(**opt_params)
        # critic_target = build_double_critic_model(self.critic_input_dim, next(self.rng))
        # critic_target = hk.transform(lambda x: DoubleCritic()(x))

        # state, action, next_state, reward, not_done = yield params

        target_Q = jax.lax.stop_gradient(
            self.get_td_target(rng=next(self.rng), params=params, **kwargs,)
        )

        params.critic, opt_params.critic, _ = self.apply_updates(
            params.critic,
            obs,
            action,
            target_Q,
            loss=self.critic_loss,
            optimizer=self.optimizer.critic,
            _params=params.critic,
            _opt_params=opt_params.critic,
        )

        # self.optimizer.critic = critic_step(
        #     optimizer=self.optimizer.critic,
        #     state=state,
        #     action=action,
        #     target_Q=target_Q,
        # )

        # if i % self.policy_freq == 0:
        # self.optimizer.actor, log_p = actor_step(
        #     rng=next(self.rng),
        #     optimizer=self.optimizer.actor,
        #     critic=self.optimizer.critic,
        #     state=state,
        #     log_alpha=self.optimizer.log_alpha,
        # )

        return vars(params), vars(opt_params)

    @functools.partial(jax.jit, static_argnums=0)
    def update_actor(self, params: dict, opt_params: dict, obs, action, **kwargs):
        params = Params(**params)
        opt_params = OptParams(**opt_params)
        params.actor, opt_params.actor, log_p = self.apply_updates(
            *vars(params).values(),
            obs,
            next(self.rng),
            loss=self.actor_loss,
            optimizer=self.optimizer.critic,
            _params=params.actor,
            _opt_params=opt_params.actor,
            has_aux=True,
        )

        if self.entropy_tune:
            # self.optimizer.log_alpha = alpha_step(
            #     optimizer=self.optimizer.log_alpha,
            #     log_p=log_p,
            #     target_entropy=self.target_entropy,
            # )
            params.log_alpha, opt_params.log_alpha, _ = self.apply_updates(
                params.log_alpha,
                log_p,
                loss=self.alpha_loss,
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

    # if load_path and i % self.save_freq == 0:
    #     save_model(load_path + "_critic", self.optimizer.critic)
    #     save_model(load_path + "_actor", self.optimizer.actor)
    #     save_model(load_path + "_log_alpha", self.optimizer.log_alpha)

    # @functools.partial(jax.jit, static_argnums=0)
    # def select_action(self, params, state):
    #     mu, _ = self.actor.apply(params, state)
    #     return mu.flatten()
    #
    @functools.partial(jax.jit, static_argnums=0)
    def step(self, params, obs, rng=None):
        mu, log_sig = self.net.actor.apply(params, obs)
        if rng is not None:  # TODO: this is gross
            mu += random.normal(rng, mu.shape) * jnp.exp(log_sig)
        return mu

    # def train(self, initial_state, replay_buffer, batch_size=100, load_path=None):
    #     if self.iterator is None:
    #         self.iterator = self.generator(initial_state, load_path=load_path)
    #         next(self.iterator)
    #
    #     data = replay_buffer.sample(next(self.rng), batch_size)
    #     return self.iterator.send(data)
