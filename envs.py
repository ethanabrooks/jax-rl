import gym
from dm_env import restart, transition, termination, specs
import dm_env
from gym.envs.classic_control import PendulumEnv, CartPoleEnv


class Environment(dm_env.Environment, gym.Wrapper):
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

    def max_action(self):
        return self.action_space.high


def register(cls, name):
    gym.register(
        id=name, entry_point=f"{cls.__module__}:{cls.__qualname__}",
    )


class PendulumEnvironment(Environment):
    def __init__(self, **kwargs):
        super().__init__(PendulumEnv(**kwargs))


class CartPoleEnvironment(Environment):
    def __init__(self):
        super().__init__(CartPoleEnv())
