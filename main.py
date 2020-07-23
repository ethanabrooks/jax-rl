import argparse
import itertools
import os
from pprint import pprint

import gym
import numpy as np
import ray
from haiku import PRNGSequence
from ray import tune
from tqdm import tqdm

import MPO
import SAC
import TD3
import configs
from envs import Environment
from levels_env import Env
from utils import ReplayBuffer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def train(kwargs, use_tune):
    _train(**kwargs, use_tune=use_tune)


def _train(
    batch_size=256,
    buffer_size=int(2e6),
    discount=0.99,
    env_id=None,
    eval_freq=5e3,
    eval_episodes=10,
    expl_noise=0.1,
    learning_rate=3e-4,
    load_model=None,
    max_time_steps=None,
    noise_clip=0.5,
    num_action_samples=20,
    policy="SAC",
    policy_freq=2,
    policy_noise=0.2,
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

    def report(**xx):
        if use_tune:
            tune.report(**xx)
        else:
            pprint(xx)

    def make_env():
        return Environment.wrap(gym.make(env_id) if env_id else Env(1000))

    def eval_policy():
        eval_env = make_env()
        eval_env.seed(seed)

        avg_reward = 0.0
        it = itertools.count() if render else tqdm(range(eval_episodes), desc="eval")
        for _ in it:
            eval_time_step = eval_env.reset()
            while not eval_time_step.last():
                if render:
                    eval_env.render()
                action = policy.select_action(eval_time_step.observation)
                eval_time_step = eval_env.step(action)
                avg_reward += eval_time_step.reward

        avg_reward /= eval_episodes

        report(eval_reward=avg_reward)
        return avg_reward

    env_name = env_id or "levels"
    file_name = f"{policy}_{env_name}_{seed}"
    report(policy=policy)
    report(env=env_name)
    report(seed=seed)
    if save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    env = make_env()
    assert isinstance(env, Environment)
    # Set seeds
    np.random.seed(seed)
    state_shape = env.observation_spec().shape
    action_dim = env.action_spec().shape[0]
    max_action = env.max_action()
    kwargs = dict(
        state_shape=state_shape,
        action_dim=action_dim,
        max_action=max_action,
        discount=discount,
        lr=learning_rate,
    )

    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs.update(
            policy_noise=policy_noise * max_action,
            noise_clip=noise_clip * max_action,
            policy_freq=policy_freq,
            expl_noise=expl_noise,
            tau=tau,
        )
        policy = TD3.TD3(**kwargs)
    elif policy == "SAC":
        kwargs.update(policy_freq=policy_freq, tau=tau)
        policy = SAC.SAC(**kwargs)
    elif policy == "MPO":
        policy = MPO.MPO(**kwargs)
    if load_model is not None:
        policy_file = file_name if load_model == "default" else load_model
        policy.load(f"./models/{policy_file}")
    replay_buffer = ReplayBuffer(state_shape, action_dim, max_size=int(buffer_size))
    # Evaluate untrained policy

    eval_policy()
    time_step = env.reset()
    episode_reward = 0
    episode_time_steps = 0
    episode_num = 0
    rng = PRNGSequence(seed)
    next(rng)

    it = range(int(max_time_steps)) if max_time_steps else itertools.count()
    for t in it:
        episode_time_steps += 1

        state = time_step.observation

        # Select action randomly or according to policy
        if t < start_time_steps:
            action = np.random.uniform(
                env.max_action(), env.min_action(), size=env.action_spec().shape,
            )
        else:
            action = (
                (policy.sample_action(next(rng), state))
                .clip(-max_action, max_action)
                .squeeze(0)
            )

        # Perform action
        time_step = env.step(action)
        done_bool = float(time_step.last())

        # Store data in replay buffer
        replay_buffer.add(
            state, action, time_step.observation, time_step.reward, done_bool
        )

        episode_reward += time_step.reward

        # Train agent after collecting sufficient data
        if t >= start_time_steps:
            for _ in range(train_steps):
                if policy == "MPO":
                    policy.train(replay_buffer, batch_size, num_action_samples)
                else:
                    policy.train(replay_buffer, batch_size)

        if time_step.last():
            # +1 to account for 0 indexing. +0 on ep_time_steps since it will increment +1 even if done=True
            report(
                time_steps=t + 1,
                episode=episode_num + 1,
                episode_time_steps=episode_time_steps,
                reward=episode_reward,
            )
            # Reset environment
            time_step = env.reset()
            episode_reward = 0
            episode_time_steps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            eval_policy()
        if (t + 1) % save_freq == 0:
            if save_model:
                save_path = f"./models/{file_name}_" + str(t + 1)
                print(f"Saving model to {save_path}")
                policy.save(save_path)


def main(config, use_tune, num_samples, local_mode, env, **kwargs):
    config = getattr(configs, config)
    config.update(env_id=env)
    if use_tune:
        ray.init(webui_host="127.0.0.1", local_mode=local_mode, **kwargs)
        metric = "reward"
        if local_mode:
            tune.run(train, config=config)
        else:
            tune.run(
                train,
                config=config,
                resources_per_trial={"gpu": 1},
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
    main(**vars(PARSER.parse_args()))
