from hyperopt import hp


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def medium_values(start, stop):
    return [2 ** i for i in range(start, stop)]


def big_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** i), 5 * (10 ** i))]


search = dict(
    actor_freq=hp.choice("actor_freq", [1, 2, 3]),
    batch_size=hp.choice("batch_size", medium_values(6, 10)),
    learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    seed=hp.randint("seed", 20),
    start_time_steps=hp.choice("start_time_steps", big_values(3, 5)),
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
    max_time_steps=big_values(3, 5),
    **search,
    **{"outer_" + k: v for k, v in search.items()}
)
double_search.update(start_time_steps=0, outer_start_time_steps=1)
