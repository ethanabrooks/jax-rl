import argparse
import itertools
import os

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
from utils import ReplayBuffer


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_id, seed, render, eval_episodes=10):
    eval_env = Environment.wrap(gym.make(env_id))
    eval_env.seed(seed)

    avg_reward = 0.0
    it = itertools.count() if render else tqdm(range(eval_episodes), desc="eval")
    for _ in it:
        time_step = eval_env.reset()
        while not time_step.last():
            if render:
                eval_env.render()
            action = policy.select_action(time_step.observation)
            time_step = eval_env.step(action)
            avg_reward += time_step.reward

    avg_reward /= eval_episodes

    tune.report(eval_reward=avg_reward)
    return avg_reward


def _train(kwargs):
    train(**kwargs)


def train(
    batch_size=256,
    buffer_size=int(2e6),
    discount=0.99,
    env_id="Pendulum-v0",
    eval_freq=5e3,
    expl_noise=0.1,
    learning_rate=3e-4,
    load_model=None,
    max_time_steps=int(1e6),
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
):
    file_name = f"{policy}_{env_id}_{seed}"
    tune.report(policy=policy)
    tune.report(env=env_id)
    tune.report(seed=seed)
    if save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    env = Environment.wrap(gym.make(env_id))
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

    # evaluations = [eval_policy(policy=policy, env_id=env_id, seed=seed, render=render)]
    eval_policy(policy=policy, env_id=env_id, seed=seed, render=render)
    time_step = env.reset()
    episode_reward = 0
    episode_time_steps = 0
    episode_num = 0
    rng = PRNGSequence(seed)
    next(rng)

    for t in range(int(max_time_steps)):
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
            tune.report(
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
            eval_policy(policy, env_id, seed, render)
        if (t + 1) % save_freq == 0:
            if save_model:
                save_path = f"./models/{file_name}_" + str(t + 1)
                print(f"Saving model to {save_path}")
                policy.save(save_path)


def main(config, local_mode):
    ray.init(local_mode=local_mode)
    tune.run(_train, config=getattr(configs, config))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("config")
    PARSER.add_argument("--local-mode", action="store_true")
    main(**vars(PARSER.parse_args()))
