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
from arguments import add_arguments
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
        batch_size,
        buffer_size,
        discount,
        env_id,
        eval_freq,
        eval_episodes,
        learning_rate,
        load_path,
        max_time_steps,
        actor_freq,
        save_freq,
        save_model,
        seed,
        start_time_steps,
        tau,
        train_steps,
        render,
        use_tune,
    ):
        seed = int(seed)
        policy = "SAC"
        self.use_tune = use_tune
        self.seed = seed
        self.max_time_steps = int(max_time_steps) if max_time_steps else None
        self.start_time_steps = int(start_time_steps)
        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.eval_freq = eval_freq

        def make_env():
            return Environment.wrap(gym.make(env_id) if env_id else Env(1000))

        def eval_policy(params):
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
        max_action = self.env.max_action()
        time_step = self.env.reset()
        iterator = self.generator(time_step)
        obs, params = next(iterator)
        for t in itertools.count():
            if t % self.eval_freq == 0:
                self.eval_policy(params)

            if t < self.start_time_steps:
                action = self.env.action_space.sample()
            else:
                action = self.policy.sample_action(next(self.rng), obs).squeeze(0)
                action = action.clip(-max_action, max_action)
            obs, params = iterator.send(action)

    def report(self, **kwargs):
        if self.use_tune:
            tune.report(**kwargs)
        else:
            pprint(kwargs)

    def generator(self, time_step):
        episode_reward = 0
        episode_time_steps = 0
        episode_num = 0

        state_shape = self.env.observation_spec().shape
        action_dim = self.env.action_spec().shape[0]

        replay_buffer = ReplayBuffer(state_shape, action_dim, max_size=self.buffer_size)
        next(self.rng)
        params, opt_params = self.policy.init(
            self.env.observation_space.sample(), self.env.action_space.sample()
        )
        for t in (
            range(int(self.max_time_steps))
            if self.max_time_steps
            else itertools.count()
        ):
            episode_time_steps += 1
            obs = time_step.observation

            # Select action randomly or according to policy
            action = yield obs, params["actor"]

            # Perform action
            time_step = self.env.step(action)
            done_bool = float(time_step.last())

            # Store data in replay buffer
            replay_buffer.add(
                obs, action, time_step.observation, time_step.reward, done_bool
            )

            episode_reward += time_step.reward

            # Train agent after collecting sufficient data
            if t >= self.start_time_steps:
                for i in range(self.train_steps):
                    data = replay_buffer.sample(next(self.rng), self.batch_size)
                    # self.policy.update_critic(
                    #     params=params, opt_params=opt_params, **vars(data)
                    # )
                    params, opt_params = self.policy.update_critic(
                        params=params, opt_params=opt_params, **vars(data),
                    )
                    if (t * self.train_steps + i) % self.policy.actor_freq == 0:
                        (params, opt_params,) = self.policy.update_actor(
                            params, opt_params, data.obs
                        )

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


def main(config, use_tune, num_samples, local_mode, env, name, **kwargs):
    config = getattr(configs, config)
    config.update(env_id=env)
    for k, v in kwargs.items():
        if k not in config:
            config[k] = v
    if use_tune:
        ray.init(webui_host="127.0.0.1", local_mode=local_mode)
        metric = "reward"
        if local_mode:
            tune.run(
                train,
                name=name,
                config=config,
                resources_per_trial={"gpu": 1, "cpu": 2},
            )
        else:
            tune.run(
                train,
                config=config,
                name=name,
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
    PARSER.add_argument("--name")
    add_arguments(PARSER)
    main(**vars(PARSER.parse_args()))
