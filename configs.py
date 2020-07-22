from ray import tune
import numpy as np


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def big_values(start, stop):
    return [2 ** i for i in range(start, stop)]


search = dict(
    learning_rate=tune.choice(small_values(2, 5)),
    batch_size=tune.choice(big_values(6, 10)),
    policy_freq=tune.choice([1, 2, 3]),
    tau=tune.choice(small_values(2, 5)),
    # seed=tune.sample_from(lambda _: np.random.randint(20)),
)

deterministic = dict(policy="SAC", max_time_steps=100000)
