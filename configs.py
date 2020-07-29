from hyperopt import hp


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def big_values(start, stop):
    return [2 ** i for i in range(start, stop)]


search = dict(
    learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    batch_size=hp.choice("batch_size", big_values(6, 10)),
    actor_freq=hp.choice("actor_freq", [1, 2, 3]),
    tau=hp.choice("tau", small_values(2, 5)),
    seed=hp.randint("seed", 20),
)

pendulum = dict(batch_size=128, learning_rate=0.01, actor_freq=1, seed=5, tau=0.005)

double = dict(
    outer_batch_size=128,
    outer_learning_rate=0.01,
    outer_actor_freq=1,
    outer_seed=5,
    outer_tau=0.005,
    batch_size=128,
    learning_rate=0.01,
    actor_freq=1,
    seed=5,
    tau=0.005,
)

double_search = dict(
    outer_learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    outer_batch_size=hp.choice("batch_size", big_values(6, 10)),
    outer_actor_freq=hp.choice("actor_freq", [1, 2, 3]),
    outer_tau=hp.choice("tau", small_values(2, 5)),
    outer_seed=hp.randint("seed", 20),
    learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    batch_size=hp.choice("batch_size", big_values(6, 10)),
    actor_freq=hp.choice("actor_freq", [1, 2, 3]),
    tau=hp.choice("tau", small_values(2, 5)),
    seed=hp.randint("seed", 20),
)
