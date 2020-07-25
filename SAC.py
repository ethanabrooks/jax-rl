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
        policy_freq=2,
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
        def actor(state, key=None):
            return GaussianPolicy(action_dim=action_dim, max_action=max_action)(
                state, key
            )

        self.net = Nets(
            actor=hk.transform(actor),
            critic=hk.without_apply_rng(
                hk.transform(lambda s, a: DoubleCritic()(s, a))
            ),
            target_critic=hk.without_apply_rng(
                hk.transform(lambda s, a: DoubleCritic()(s, a))
            ),
            log_alpha=hk.without_apply_rng(
                hk.transform(lambda: Constant()(initial_log_alpha))
            ),
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
            **{k: optix.adam(learning_rate=lr) for k in vars(self.net).keys()}
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.save_freq = save_freq

        self.total_it = 0

    def init(self, state, action):
        key = next(self.rng)
        critic_params = self.net.critic.init(key, state, action)
        params = Params(
            actor=self.net.actor.init(key, state, key=None),
            critic=critic_params,
            target_critic=critic_params,
            log_alpha=self.net.log_alpha.init(key),
        )

        def opt_params():
            for (k1, param), (k2, optimizer) in zip(vars(params), vars(self.optimizer)):
                assert k1 == k2
                yield optimizer.init(param)

        return params, Params(*opt_params())

    # @jax.jit
    @jax.lax.stop_gradient
    def get_td_target(
        self,
        rng: PRNGKey,
        params: Params,
        opt_params: Params,
        next_state: jnp.ndarray,
        reward: float,
        not_done: bool,
    ):
        next_action, next_log_p = self.net.actor.apply(
            params.actor, next_state, sample=True, key=rng
        )

        target_Q1, target_Q2 = self.net.target_critic.apply(
            params.target_critic, next_state, next_action
        )
        target_Q = (
            jnp.minimum(target_Q1, target_Q2)
            - jnp.exp(self.net.log_alpha.apply()) * next_log_p
        )
        target_Q = reward + not_done * self.discount * target_Q

        return target_Q

    # noinspection PyPep8Naming
    def critic_loss(
        self, params: jnp.ndarray, state, action, target_Q: jnp.ndarray,
    ):
        current_Q1, current_Q2 = self.net.critic.apply(params, state, action)
        critic_loss = double_mse(current_Q1, current_Q2, target_Q)
        return jnp.mean(critic_loss)

    def actor_loss(self, state, key: PRNGKey, params: Params):
        actor_action, log_p = self.net.actor.apply(
            params.actor, state, sample=True, key=key
        )
        q1, q2 = self.net.critic.apply(params.critic, state, actor_action)
        min_q = jnp.minimum(q1, q2)
        partial_loss_fn = jax.vmap(
            partial(
                actor_loss_fn,
                jax.lax.stop_gradient(self.net.log_alpha.apply(params.log_alpha)),
            )
        )
        actor_loss = partial_loss_fn(log_p, min_q)
        return jnp.mean(actor_loss), log_p

    def alpha_loss(self, log_p):
        partial_loss_fn = jax.vmap(
            partial(alpha_loss_fn, self.net.log_alpha.apply(), self.target_entropy)
        )
        return jnp.mean(partial_loss_fn(log_p))

    def update(
        self, i: int, params: Params, opt_params: Params, state, action, **kwargs
    ):
        # critic_target = build_double_critic_model(self.critic_input_dim, next(self.rng))
        # critic_target = hk.transform(lambda x: DoubleCritic()(x))

        # state, action, next_state, reward, not_done = yield params

        target_Q = self.get_td_target(rng=next(self.rng), params=params, **kwargs,)

        def apply_updates(loss, _params, _opt_params, **_kwargs):
            grad = jax.grad(loss)(**_kwargs)
            updates, _opt_params = self.optimizer.critic.update(grad, _opt_params)
            return optix.apply_updates(_params, updates), _opt_params

        params.critic, opt_params.critic = apply_updates(
            loss=self.critic_loss,
            _params=params.critic,
            _opt_params=opt_params.critic,
            params=params,
            state=state,
            action=action,
            target_Q=target_Q,
        )

        # self.optimizer.critic = critic_step(
        #     optimizer=self.optimizer.critic,
        #     state=state,
        #     action=action,
        #     target_Q=target_Q,
        # )

        if i % self.policy_freq == 0:
            # self.optimizer.actor, log_p = actor_step(
            #     rng=next(self.rng),
            #     optimizer=self.optimizer.actor,
            #     critic=self.optimizer.critic,
            #     state=state,
            #     log_alpha=self.optimizer.log_alpha,
            # )

            params.actor, opt_params.actor = apply_updates(
                loss=self.actor_loss,
                _params=params.actor,
                _opt_params=opt_params.actor,
                state=state,
                key=next(self.rng),
                params=params,
            )

            if self.entropy_tune:
                # self.optimizer.log_alpha = alpha_step(
                #     optimizer=self.optimizer.log_alpha,
                #     log_p=log_p,
                #     target_entropy=self.target_entropy,
                # )
                params.log_alpha, opt_params.log_alpha = apply_updates(
                    loss=self.alpha_loss,
                    _params=params.log_alpha,
                    _opt_params=opt_params.log_alpha,
                    log_p=log_p,
                )

            params.target_critic = jax.tree_multimap(
                lambda p1, p2: self.tau * p1 + (1 - self.tau) * p2,
                params.target_critic,
                params.critic,
            )

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
    def step(self, params, state, rng=None):
        mu, log_sig = self.net.actor.apply(params, state)
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
