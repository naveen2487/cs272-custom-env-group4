import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
import numpy as np


def train_stock_trading_env():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the custom environment
    from ray.tune.registry import register_env
    def env_creator(env_config):
        # stock_data = np.random.rand(100, 5) * 100  # Dummy stock data
        return StockTradingEnv(csv_file="/content/MSFT.csv")
    register_env("StockTradingEnv", env_creator)

    # Configure the training algorithm
    config = {
        "env": "StockTradingEnv",
        "num_gpus": 0,
        "num_workers": 1,
        "framework": "tf",
        "lr": 1e-4,  # Adjust learning rate
        "gamma": 0.99,
        "vf_clip_param": 50.0,
        "env_config": {},
    }

    # Initialize PPO algorithm with the given configuration
    algo = PPO(config=config)

    # Train the model
    for i in range(50):  # Number of training iterations
        result = algo.train()
        if (i+1) % 5 ==0:
          checkpoint_dir = algo.save().checkpoint.path
        print(f"Iteration: {i}, reward: {result['episode_reward_mean']}")

    # Shut down Ray
    ray.shutdown()

# Run the training
train_stock_trading_env()
