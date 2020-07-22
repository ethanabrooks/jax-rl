from ray import tune
import numpy as np
from hyperopt import hp


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def big_values(start, stop):
    return [2 ** i for i in range(start, stop)]


search = dict(
    learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    batch_size=hp.choice("batch_size", big_values(6, 10)),
    policy_freq=hp.choice("policy_freq", [1, 2, 3]),
    tau=hp.choice("tau", small_values(2, 5)),
    seed=hp.randint("seed", 20),
)

deterministic = dict(policy="SAC", max_time_steps=100000)
