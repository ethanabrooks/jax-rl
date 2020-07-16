from ray import tune
import numpy as np

search_space = dict(
    lr=tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    momentum=tune.uniform(0.1, 0.9),
)

deterministic = dict(policy="SAC", max_time_steps=100000)
