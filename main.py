import argparse
import itertools
import os
from pprint import pprint

import gym
import numpy as np
import ray
from haiku import PRNGSequence
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from tqdm import tqdm

import SAC
import configs
from envs import Environment
from levels_env import Env
from utils import ReplayBuffer


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def train(kwargs, use_tune):
    Trainer(**kwargs, use_tune=use_tune).train()


class Trainer:
    def __init__(
        self,
        batch_size=256,
        buffer_size=int(2e6),
        discount=0.99,
        env_id=None,
        eval_freq=5e3,
        eval_episodes=10,
        learning_rate=3e-4,
        load_path=None,
        max_time_steps=None,
        actor_freq=2,
        save_freq=int(5e3),
        save_model=True,
        seed=0,
        start_time_steps=int(1e4),
        tau=0.005,
        train_steps=1,
        render=False,
        use_tune=True,
    ):
        seed = int(seed)
        policy = "SAC"
        self.use_tune = use_tune
        self.seed = seed
        self.max_time_steps = max_time_steps
        self.start_time_steps = start_time_steps
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.eval_freq = eval_freq

        def make_env():
            return Environment.wrap(gym.make(env_id) if env_id else Env(1000))

        def eval_policy():
            eval_env = make_env()
            eval_env.seed(seed)

            avg_reward = 0.0
            it = (
                itertools.count() if render else tqdm(range(eval_episodes), desc="eval")
            )
            for _ in it:
                eval_time_step = eval_env.reset()
                while not eval_time_step.last():
                    if render:
                        eval_env.render()
                    action = policy.select_action(eval_time_step.observation)
                    eval_time_step = eval_env.step(action)
                    avg_reward += eval_time_step.reward

            avg_reward /= eval_episodes

            self.report(eval_reward=avg_reward)
            return avg_reward

        self.eval_policy = eval_policy

        env_name = env_id or "levels"
        file_name = f"{policy}_{env_name}_{seed}"
        self.report(policy=policy)
        self.report(env=env_name)
        self.report(seed=seed)
        if save_model and not os.path.exists("./models"):
            os.makedirs("./models")
        self.env = env = make_env()
        assert isinstance(env, Environment)
        # Set seeds
        np.random.seed(seed)
        state_shape = env.observation_spec().shape
        action_dim = env.action_spec().shape[0]
        max_action = env.max_action()
        # Initialize policy
        self.policy = policy = SAC.SAC(  # TODO
            state_shape=state_shape,
            action_dim=action_dim,
            max_action=max_action,
            save_freq=save_freq,
            discount=discount,
            lr=learning_rate,
            actor_freq=actor_freq,
            tau=tau,
        )
        self.rng = PRNGSequence(self.seed)

    def train(self):
        iterator = self.generator()
        max_action = self.env.max_action()
        state = next(iterator)
        for t in itertools.count():
            if t % self.eval_freq == 0:
                self.eval_policy()

            if t < self.start_time_steps:
                action = self.env.action_space.sample()
            else:
                action = (
                    (self.policy.sample_action(next(self.rng), state))
                    .clip(-max_action, max_action)
                    .squeeze(0)
                )
            state = iterator.send(action)

    def report(self, **kwargs):
        if self.use_tune:
            tune.report(**kwargs)
        else:
            pprint(kwargs)

    def generator(self):
        time_step = self.env.reset()
        self.policy._init()
        episode_reward = 0
        episode_time_steps = 0
        episode_num = 0

        state_shape = self.env.observation_spec().shape
        action_dim = self.env.action_spec().shape[0]
        replay_buffer = ReplayBuffer(state_shape, action_dim, max_size=self.buffer_size)
        next(self.rng)

        for t in (
            range(int(self.max_time_steps))
            if self.max_time_steps
            else itertools.count()
        ):
            episode_time_steps += 1
            state = time_step.observation

            # Select action randomly or according to policy
            action = yield state

            # Perform action
            time_step = self.env.step(action)
            done_bool = float(time_step.last())

            # Store data in replay buffer
            replay_buffer.add(
                state, action, time_step.observation, time_step.reward, done_bool
            )

            episode_reward += time_step.reward

            # Train agent after collecting sufficient data
            if t >= self.start_time_steps:
                for _ in range(self.train_steps):
                    self.policy.train(replay_buffer, self.batch_size)

            if time_step.last():
                # +1 to account for 0 indexing. +0 on ep_time_steps since it will increment +1 even if done=True
                self.report(
                    time_steps=t + 1,
                    episode=episode_num + 1,
                    episode_time_steps=episode_time_steps,
                    reward=episode_reward,
                )
                # Reset environment
                time_step = self.env.reset()
                episode_reward = 0
                episode_time_steps = 0
                episode_num += 1


def main(config, use_tune, num_samples, local_mode, env, load_path):
    config = getattr(configs, config)
    config.update(env_id=env, load_path=load_path)
    if use_tune:
        ray.init(webui_host="127.0.0.1", local_mode=local_mode)
        metric = "reward"
        if local_mode:
            tune.run(train, config=config, resources_per_trial={"gpu": 1, "cpu": 2})
        else:
            tune.run(
                train,
                config=config,
                resources_per_trial={"gpu": 1, "cpu": 2},
                # scheduler=ASHAScheduler(metric=metric, mode="max"),
                search_alg=HyperOptSearch(config, metric=metric, mode="max"),
                num_samples=num_samples,
            )
    else:
        train(config, use_tune=use_tune)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("config")
    PARSER.add_argument("--no-tune", dest="use_tune", action="store_false")
    PARSER.add_argument("--local-mode", action="store_true")
    PARSER.add_argument("--num-samples", type=int)
    PARSER.add_argument("--env")
    PARSER.add_argument("--load-path")
    main(**vars(PARSER.parse_args()))
