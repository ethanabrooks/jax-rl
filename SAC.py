import functools
from dataclasses import dataclass
from functools import partial
from typing import Union

import flax.nn as nn
import jax
import jax.numpy as jnp
from flax import optim
from flax.optim import Adam, Optimizer
from haiku import PRNGSequence
from jax import random

from models import (
    GaussianPolicy,
    DoubleCritic,
    Constant,
)
from utils import double_mse, apply_model, copy_params
from utils import gaussian_likelihood


@dataclass
class Optimizers:
    T = Union[optim.Adam, optim.Optimizer]
    actor: T
    critic: T
    log_alpha: T


@dataclass
class Models:
    T = nn.Model
    actor: T
    critic: T
    target_critic: T
    alpha: T


@dataclass
class Modules:
    T = nn.Module
    actor: T
    critic: T
    alpha: T


def actor_loss_fn(log_p, min_q):
    return (log_p - min_q).mean()


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
        entropy_tune=False,
        seed=0,
    ):

        self.rng = PRNGSequence(seed)

        actor_input_dim = [((1, *state_shape), jnp.float32)]
        critic_input_dim = self.critic_input_dim = [
            ((1, *state_shape), jnp.float32),
            ((1, action_dim), jnp.float32),
        ]
        self.actor = None
        self.critic = None
        self.log_alpha = None
        self.entropy_tune = entropy_tune
        self.target_entropy = -action_dim

        self.adam = Optimizers(
            actor=optim.Adam(learning_rate=lr),
            critic=optim.Adam(learning_rate=lr),
            log_alpha=optim.Adam(learning_rate=lr),
        )
        self.module = Modules(
            actor=GaussianPolicy.partial(action_dim=action_dim, max_action=max_action),
            critic=DoubleCritic.partial(),
            alpha=Constant.partial(start_value=1),
        )
        self.optimizer = None

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = actor_freq
        self.save_freq = save_freq

        self.total_it = 0
        self.model = None

        def new_params(module: nn.Module, shape=None):
            _, params = (
                module.init(next(self.rng))
                if shape is None
                else module.init_by_shape(next(self.rng), shape)
            )
            return params

        def new_model(module: nn.Module, shape=None) -> nn.Model:
            return nn.Model(module, new_params(module, shape))

        def update_model(model: nn.Model, shape=None) -> nn.Model:
            return model.replace(params=new_params(model.module, shape))

        def reset_models() -> Models:
            if self.model is None:
                critic = new_model(self.module.critic, critic_input_dim)
                return Models(
                    actor=new_model(self.module.actor, actor_input_dim),
                    critic=critic,
                    target_critic=critic.replace(params=critic.params),
                    alpha=new_model(self.module.alpha),
                )
            else:
                critic = update_model(self.model.critic, critic_input_dim)
                return Models(
                    actor=update_model(self.model.actor, actor_input_dim),
                    critic=critic,
                    target_critic=critic.replace(params=critic.params),
                    alpha=update_model(self.model.alpha),
                )

        self.reset_models = reset_models

        def reset_optimizer(adam: Adam, model: nn.Model) -> Optimizer:
            return jax.device_put(adam.create(model))

        def reset_optimizers() -> Optimizers:
            return Optimizers(
                actor=reset_optimizer(self.adam.actor, self.model.actor),
                critic=reset_optimizer(self.adam.critic, self.model.critic),
                log_alpha=reset_optimizer(self.adam.log_alpha, self.model.alpha),
            )

        self.reset_optimizers = reset_optimizers
        self.i = 0

    def init(self):
        self.model = self.reset_models()
        self.optimizer = self.reset_optimizers()
        self.i = 0
        # if load_path:
        #     self.optimizer = Optimizers(
        #         actor=load_model(load_path + "_actor", self.optimizer.actor),
        #         critic=load_model(load_path + "_critic", self.optimizer.critic),
        #         log_alpha=load_model(
        #             load_path + "_log_alpha", self.optimizer.log_alpha
        #         ),
        #     )
        #     critic_target = critic_target.replace(
        #         params=self.optimizer.critic.target.params
        #     )

    def update(self, obs, action, **kwargs):
        self.i += 1
        target_Q = jax.lax.stop_gradient(
            self.get_td_target(
                **kwargs,
                discount=self.discount,
                actor=self.optimizer.actor.target,
                critic_target=(self.model.target_critic),
                log_alpha=self.optimizer.log_alpha.target
            )
        )

        self.optimizer.critic = self.critic_step(
            optimizer=self.optimizer.critic,
            state=obs,
            action=action,
            target_Q=target_Q,
        )

        if self.i % self.policy_freq == 0:
            self.optimizer.actor, log_p = self.actor_step(
                rng=next(self.rng),
                optimizer=self.optimizer.actor,
                critic=self.optimizer.critic,
                state=obs,
                log_alpha=self.optimizer.log_alpha,
            )

            if self.entropy_tune:
                self.optimizer.log_alpha = self.alpha_step(
                    optimizer=self.optimizer.log_alpha,
                    log_p=log_p,
                    target_entropy=self.target_entropy,
                )

            self.model.target_critic = copy_params(
                self.optimizer.critic.target, self.model.target_critic, self.tau
            )
        # if load_path and i % self.save_freq == 0:
        #     save_model(load_path + "_critic", self.optimizer.critic)
        #     save_model(load_path + "_actor", self.optimizer.actor)
        #     save_model(load_path + "_log_alpha", self.optimizer.log_alpha)

    def select_action(self, state):
        mu, _ = apply_model(self.optimizer.actor.target, state)
        return mu.flatten()

    def sample_action(self, rng, state):
        mu, log_sig = apply_model(self.optimizer.actor.target, state)
        return mu + random.normal(rng, mu.shape) * jnp.exp(log_sig)

    @functools.partial(jax.jit, static_argnums=0)
    def get_td_target(
        self, next_obs, reward, not_done, discount, actor, critic_target, log_alpha,
    ):
        mu, _ = actor(next_obs)
        next_action = 2 * nn.tanh(mu)

        target_Q1, target_Q2 = critic_target(next_obs, next_action)
        target_Q = jnp.minimum(target_Q1, target_Q2)
        target_Q = reward + not_done * discount * target_Q

        return target_Q

    @functools.partial(jax.jit, static_argnums=0)
    def critic_step(self, optimizer, state, action, target_Q):
        def loss_fn(critic):
            current_Q1, current_Q2 = critic(state, action)
            critic_loss = double_mse(current_Q1, current_Q2, target_Q)
            return jnp.mean(critic_loss)

        grad = jax.grad(loss_fn)(optimizer.target)
        return optimizer.apply_gradient(grad)

    @functools.partial(jax.jit, static_argnums=0)
    def actor_step(self, rng, optimizer, critic, state, log_alpha):
        critic, log_alpha = critic.target, log_alpha.target

        def loss_fn(actor):
            mu, log_sig = actor(state, key=rng)
            pi = mu + random.normal(rng, mu.shape) * jnp.exp(log_sig)
            log_p = gaussian_likelihood(pi, mu, log_sig)
            pi = nn.tanh(pi)
            log_p -= jnp.sum(jnp.log(nn.relu(1 - pi ** 2) + 1e-6), axis=1)
            actor_action = 2 * pi

            q1, q2 = critic(state, actor_action)
            min_q = jnp.minimum(q1, q2)
            partial_loss_fn = jax.vmap(partial(actor_loss_fn))
            actor_loss = partial_loss_fn(log_p, min_q)
            return jnp.mean(actor_loss), log_p

        grad, log_p = jax.grad(loss_fn, has_aux=True)(optimizer.target)
        return optimizer.apply_gradient(grad), log_p

    @functools.partial(jax.jit, static_argnums=0)
    def alpha_step(self, optimizer, log_p, target_entropy):
        log_p = jax.lax.stop_gradient(log_p)

        def loss_fn(log_alpha):
            partial_loss_fn = jax.vmap(
                partial(alpha_loss_fn, log_alpha(), target_entropy)
            )
            return jnp.mean(partial_loss_fn(log_p))

        grad = jax.grad(loss_fn)(optimizer.target)
        return optimizer.apply_gradient(grad)
