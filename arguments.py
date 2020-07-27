def add_arguments(parser):
    parser.add_argument(
        "--batch-size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument(
        "--buffer-size", default=2e6, type=int
    )  # Max size of replay buffer
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument(
        "--env", default="Pendulum-v0"
    )  # DeepMind control suite environment name
    parser.add_argument(
        "--eval-freq", default=5e3, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--eval-episodes", default=10, type=int
    )  # How often (time steps) we evaluate
    # parser.add_argument(
    #     "--expl-noise", default=0.1
    # )  # Std of Gaussian exploration noise
    parser.add_argument(
        "--learning-rate", default=3e-4, type=float
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--load-path", default=None
    )  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument(
        "--max-time-steps", default=1e6, type=int
    )  # Max time steps to run environment
    # parser.add_argument(
    #     "--noise-clip", default=0.5
    # )  # Range to clip target policy noise
    # parser.add_argument("--num-action-samples", default=20, type=int)
    # parser.add_argument("--policy", default="SAC")  # Policy name (TD3, SAC, or MPO)
    parser.add_argument(
        "--actor-freq", default=2, type=int
    )  # Frequency of delayed policy updates
    # parser.add_argument(
    #     "--policy-noise", default=0.2
    # )  # Noise added to target policy during critic update
    parser.add_argument(
        "--render", action="store_true"
    )  # Save model and optimizer parameters
    parser.add_argument("--save-freq", default=5e3, type=int)
    parser.add_argument(
        "--save-model", action="store_true"
    )  # Save model and optimizer parameters
    parser.add_argument("--seed", default=0, type=int)  # Sets DM control and JAX seeds
    parser.add_argument(
        "--start-time-steps", default=1e4, type=int
    )  # Time steps initial random policy is used
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--train-steps", default=1, type=int)
    # parser.add_argument(
    #     "--actor-updates", default=1, type=int
    # )  # Number of gradient steps for policy network per update
