import abc
from abc import ABC

import dm_env
import gym
from dm_env import restart, transition, termination, specs


class Environment(dm_env.Environment, gym.Wrapper):
    @staticmethod
    def wrap(env: gym.Env):
        if isinstance(env.action_space, gym.spaces.Box):
            return ContinuousActionEnvironment(env)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            return DiscreteActionEnvironment(env)

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

    def action_spec(self):
        return specs.Array(
            self.action_space.shape, dtype=self.observation_space.dtype, name="action",
        )

    @abc.abstractmethod
    def max_action(self):
        raise NotImplementedError

    @abc.abstractmethod
    def min_action(self):
        raise NotImplementedError


class ContinuousActionEnvironment(Environment):
    def max_action(self):
        assert isinstance(self.action_space, gym.spaces.Box)
        return self.action_space.high

    def min_action(self):
        assert isinstance(self.action_space, gym.spaces.Box)
        return self.action_space.low


class DiscreteActionEnvironment(Environment):
    def max_action(self):
        assert isinstance(self.action_space, gym.spaces.Discrete)
        return (self.action_space.n,)

    def min_action(self):
        return (0,)


class DiscreteObservationEnvironment(Environment, ABC):
    def observation_spec(self):
        return specs.Array((1,), dtype=int, name="observation")
