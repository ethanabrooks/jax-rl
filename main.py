import argparse
import os

import gym
import numpy as np
from tqdm import tqdm

import MPO
import SAC
import TD3
from envs import Environment
from utils import ReplayBuffer


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_id, seed, eval_episodes=10):
    gym.make("Pendulum-v0")
    eval_env = gym.make(env_id)
    eval_env.seed(seed)

    avg_reward = 0.0
    for _ in tqdm(range(eval_episodes), desc="eval"):
        time_step = eval_env.reset()
        while not time_step.last():
            action = policy.select_action(time_step.observation)
            time_step = eval_env.step(action)
            avg_reward += time_step.reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def add_arguments():
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, SAC, or MPO)
    parser.add_argument(
        "--env", dest="env_id", default="Pendulum-v1"
    )  # DeepMind control suite environment name
    parser.add_argument("--train-steps", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)  # Sets DM control and JAX seeds
    parser.add_argument(
        "--start-time-steps", default=1e4, type=int
    )  # Time steps initial random policy is used
    parser.add_argument(
        "--buffer-size", default=2e6, type=int
    )  # Max size of replay buffer
    parser.add_argument(
        "--eval-freq", default=5e3, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--max-time-steps", default=1e6, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--expl-noise", default=0.1
    )  # Std of Gaussian exploration noise
    parser.add_argument(
        "--batch-size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument(
        "--policy-noise", default=0.2
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--noise-clip", default=0.5
    )  # Range to clip target policy noise
    parser.add_argument(
        "--policy-freq", default=2, type=int
    )  # Frequency of delayed policy updates
    # parser.add_argument(
    #     "--actor-updates", default=1, type=int
    # )  # Number of gradient steps for policy network per update
    parser.add_argument(
        "--save-model", action="store_true"
    )  # Save model and optimizer parameters
    parser.add_argument(
        "--load-model", default=""
    )  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--num-action-samples", default=20, type=int)
    parser.add_argument("--save-freq", default=5e3, type=int)


def main(
    batch_size,
    buffer_size,
    discount,
    env_id,
    eval_freq,
    expl_noise,
    load_model,
    max_time_steps,
    noise_clip,
    num_action_samples,
    policy,
    policy_freq,
    policy_noise,
    save_freq,
    save_model,
    seed,
    start_time_steps,
    tau,
    train_steps,
):
    file_name = f"{policy}_{env_id}_{seed}"
    print("---------------------------------------")
    print(f"Policy: {policy}, Env: {env_id}, Seed: {seed}")
    print("---------------------------------------")
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")
    env = gym.make(env_id)
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
    if load_model != "":
        policy_file = file_name if load_model == "default" else load_model
        policy.load(f"./models/{policy_file}")
    replay_buffer = ReplayBuffer(state_shape, action_dim, max_size=int(buffer_size))
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env_id, seed)]
    time_step = env.reset()
    episode_reward = 0
    episode_time_steps = 0
    episode_num = 0
    for t in range(int(max_time_steps)):

        episode_time_steps += 1

        state = time_step.observation

        # Select action randomly or according to policy
        if t < start_time_steps:
            action = np.random.uniform(
                env.max_action(), env.min_action(), size=env.action_spec().shape,
            )
        else:
            action = (policy.select_action(state)).clip(-max_action, max_action)

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
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_time_steps} Reward: "
                f"{episode_reward:.3f}"
            )
            # Reset environment
            time_step = env.reset()
            episode_reward = 0
            episode_time_steps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval_policy(policy, env_id, seed))
            np.save(f"./results/{file_name}", evaluations)
        if (t + 1) % save_freq == 0:
            if save_model:
                policy.save(f"./models/{file_name}_" + str(t + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments()
    main(**vars(parser.parse_args()))
