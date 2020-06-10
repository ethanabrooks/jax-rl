import abc
from typing import Tuple

import gym
import jax.numpy as jnp
import numpy as np
from dm_env import restart, transition, termination, specs
from flax import nn
from jax import random

from utils import gaussian_likelihood


class Environment(gym.Wrapper):
    def reset(self):
        reset = self.env.reset()
        return restart(reset)

    def step(self, u):
        s, r, t, i = self.env.step(u)
        return termination(r, s) if t else transition(r, s)

    def observation_spec(self):
        return specs.Array(
            self.observation_space.shape,
            dtype=self.observation_space.dtype,
            name="observation",
        )

    @property
    def observation_shape(self) -> Tuple[int]:
        return self.observation_space.shape

    @property
    @abc.abstractmethod
    def actor_dim(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_action(
        self, x, key=None, sample=False,
    ):
        raise NotImplementedError


class ContinuousActionEnvironment(Environment):
    @property
    def actor_dim(self) -> int:
        return 2 * int(np.prod(self.action_space.shape))

    @property
    def action_dim(self) -> int:
        return int(np.prod(self.action_space.shape))

    def sample_action(self, x, key=None, sample=False):
        _, d = x.shape
        assert d == self.actor_dim
        log_sig_min = -20
        log_sig_max = 2
        mu, log_sig = jnp.split(x, 2, axis=-1)
        log_sig = nn.softplus(log_sig)
        log_sig = jnp.clip(log_sig, log_sig_min, log_sig_max)
        min_action = self.action_space.low
        max_action = self.action_space.high

        if not sample:
            return max_action * nn.tanh(mu), log_sig
        else:
            sig = jnp.exp(log_sig)
            pi = mu + random.normal(key, mu.shape) * sig
            log_pi = gaussian_likelihood(pi, mu, log_sig)
            log_pi -= jnp.sum(jnp.log(nn.relu(1 - nn.tanh(pi) ** 2) + 1e-6), axis=1)
            pi = nn.sigmoid(pi) * (max_action - min_action) + min_action
            return pi, log_pi


class DiscreteActionEnvironment(Environment):
    @property
    def actor_dim(self) -> int:
        return self.action_space.n

    @property
    def action_dim(self) -> int:
        return self.action_space.n

    def sample_action(self, x, key=None, sample=False, **kwargs):
        _, d = x.shape
        assert d == self.actor_dim
        if not sample:
            return jnp.argmax(x, axis=-1)
        else:
            pi = nn.softmax(x)
            choice = random.categorical(key, x)
            return (choice, pi[choice])


# class DiscreteObservationEnvironment(Environment, ABC): raise NotImplementedError


def register(cls, name):
    gym.register(
        id=name, entry_point=f"{cls.__module__}:{cls.__qualname__}",
    )


class PendulumEnvironment(ContinuousActionEnvironment):
    def __init__(self, **kwargs):
        super().__init__(gym.make("Pendulum-v0", **kwargs))


class CartPoleEnvironment(DiscreteActionEnvironment):
    def __init__(self):
        super().__init__(gym.make("CartPole-v1"))


register(CartPoleEnvironment, "CartPole-v2")
register(PendulumEnvironment, "Pendulum-v1")
