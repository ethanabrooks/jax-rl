import gym
from gym.spaces import Box
import numpy as np
import numpy as np


class Env(gym.Env):
    def __init__(self, n):
        self._render = None
        self.observation_space = Box(low=np.zeros(2), high=n * np.ones(2))
        self.action_space = Box(low=np.zeros(1), high=np.ones(1))
        self.n = n
        self.iterator = None

    def step(self, action):
        s, r, t, i = np.array(self.iterator.send(action))
        return np.array(s), r, t, i

    def reset(self):
        self.iterator = self.generator()
        s, r, t, i = np.array(next(self.iterator))
        return np.array(s)

    def generator(self):
        action = None
        for i in range(self.n):
            yield (i, i), i, False, {}
            for j in range(i):

                def render():
                    print(i, j, action)

                self._render = render
                action = yield (i, j), -1, False, {}
                if np.random.random() < action:
                    yield (i, j), i, True, {}
        yield (self.n, 0), self.n, True, {}

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input()
