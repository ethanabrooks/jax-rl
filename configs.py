from hyperopt import hp


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def big_values(start, stop):
    return [2 ** i for i in range(start, stop)]


search = dict(
    actor_freq=hp.choice("actor_freq", [1, 2, 3]),
    batch_size=hp.choice("batch_size", big_values(6, 10)),
    learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    seed=hp.randint("seed", 20),
    start_time_steps=hp.choice("start_time_steps", big_values(3, 6)),
    tau=hp.choice("tau", small_values(2, 5)),
    train_steps=hp.choice("train_steps", [1, 2, 3]),
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
    outer_actor_freq=hp.choice("actor_freq", [1, 2, 3]),
    outer_batch_size=hp.choice("batch_size", big_values(6, 10)),
    outer_learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    outer_seed=hp.randint("seed", 20),
    outer_start_time_steps=hp.choice("start_time_steps", big_values(3, 6)),
    outer_tau=hp.choice("tau", small_values(2, 5)),
    outer_train_steps=hp.choice("train_steps", [1, 2, 3]),
    actor_freq=hp.choice("actor_freq", [1, 2, 3]),
    batch_size=hp.choice("batch_size", big_values(6, 10)),
    learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    seed=hp.randint("seed", 20),
    start_time_steps=hp.choice("start_time_steps", big_values(3, 6)),
    tau=hp.choice("tau", small_values(2, 5)),
    train_steps=hp.choice("train_steps", [1, 2, 3]),
)
