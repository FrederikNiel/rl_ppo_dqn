import gymnasium as gym

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import DQN


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100000,
    "env_id": gym.make("CartPole-v1", render_mode="rgb_array"),
}
run = wandb.init(
    project="lecture9-cartpole-dqn",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

env = gym.make("CartPole-v1", render_mode="human")

model = DQN(config["policy_type"],
            config["env_id"],
            verbose=1,
            tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
model.save("dqn_cartpole")

del model  # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

wandb.finish()
