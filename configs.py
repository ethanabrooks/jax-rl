from hyperopt import hp
from pathlib import Path
import json


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def medium_values(start, stop):
    return [2 ** i for i in range(start, stop)]


def big_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** i), 5 * (10 ** i))]


def get_config(name):
    path = Path("configs", name).with_suffix(".json")
    if path.exists():
        with path.open() as f:
            config = json.load(f)
            del config["use_tune"]
            return config
    return configs[name]


search = dict(
    actor_freq=hp.choice("actor_freq", [1, 2, 3]),
    batch_size=hp.choice("batch_size", medium_values(6, 10)),
    learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    seed=hp.randint("seed", 20),
    start_time_steps=hp.choice("start_time_steps", big_values(3, 4)),
    tau=hp.choice("tau", small_values(2, 5)),
    train_steps=hp.choice("train_steps", [1, 2, 3]),
)

pendulum = dict(batch_size=128, learning_rate=0.01, actor_freq=1, seed=5, tau=0.005)

double = dict(
    outer_batch_size=128,
    outer_learning_rate=0.01,
    outer_actor_freq=1,
    outer_eval_freq=None,
    outer_seed=5,
    outer_tau=0.005,
    batch_size=128,
    learning_rate=0.01,
    actor_freq=1,
    seed=5,
    tau=0.005,
    start_time_steps=0,
    outer_start_time_steps=1,
    max_time_steps=10,
)

double_search = dict(
    max_time_steps=hp.choice("max_time_steps", big_values(3, 5)),
    outer_eval_freq=None,
    **search,
    **{"outer_" + k: v for k, v in search.items()}
)
double_search.update(start_time_steps=0, outer_start_time_steps=1)

configs = dict(
    search=search, pendulum=pendulum, double=double, double_search=double_search,
)
