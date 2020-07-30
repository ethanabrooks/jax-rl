import argparse
import re
from functools import partial

from arguments import add_arguments
from envs import Environment
from main import Trainer
from trainer_env import TrainerEnv


class OuterTrainer(Trainer):
    @classmethod
    def run(cls, config):
        outer = re.compile(r"^outer_(.*)")

        def get_kwargs(condition, f=None):
            for k, v in config.items():
                if condition(k):
                    yield k if f is None else f(k), v

        trainer_kwargs = dict(
            get_kwargs(outer.match, partial(outer.sub, r"\1")),
            **dict(
                env_id=None,
                load_path=None,
                max_time_steps=None,
                render=False,
                save_freq=None,
                save_model=False,
                use_tune=config["use_tune"],
            )
        )
        env_kwargs = dict(get_kwargs(lambda k: not outer.match(k)))
        env = Environment.wrap(TrainerEnv(**env_kwargs))
        cls(**trainer_kwargs, env=env, prefix="outer").train()


def add_outer_arguments(parser):
    parser.add_argument(
        "--outer-batch-size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument(
        "--outer-buffer-size", default=2e6, type=int
    )  # Max size of replay buffer
    parser.add_argument("--outer-discount", default=0.99)  # Discount factor
    parser.add_argument(
        "--outer-eval-freq", default=5e3, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--outer-eval-episodes", default=10, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--outer-learning-rate", default=3e-4, type=float
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--outer-actor-freq", default=2, type=int
    )  # Frequency of delayed policy updates
    parser.add_argument(
        "--outer-seed", default=0, type=int
    )  # Sets DM control and JAX seeds
    parser.add_argument(
        "--outer-start-time-steps", default=1e4, type=int
    )  # Time steps initial random policy is used
    parser.add_argument("--outer-tau", default=0.005)  # Target network update rate
    parser.add_argument("--outer-train-steps", default=1, type=int)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("config")
    PARSER.add_argument("--no-tune", dest="use_tune", action="store_false")
    PARSER.add_argument("--local-mode", action="store_true")
    PARSER.add_argument("--num-samples", type=int)
    PARSER.add_argument("--name")
    OUTER_PARSER = PARSER.add_argument_group("outer")
    add_outer_arguments(OUTER_PARSER)
    INNER_PARSER = PARSER.add_argument_group("inner")
    add_arguments(INNER_PARSER)
    OuterTrainer.main(**vars(PARSER.parse_args()))
