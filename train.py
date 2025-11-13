# =========================================================
# train_all.py
# =========================================================
# Train multiple DRL models (PPO, A2C, DQN)
# for Cloud Resource Allocation in CloudEnv
# =========================================================

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from cloud_env import CloudEnv
import os

# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------
def make_env():
    return CloudEnv(n_pms=8, n_vms=20, max_steps=200)

env = DummyVecEnv([make_env])

# ---------------------------------------------------------
# Algorithm configurations
# ---------------------------------------------------------
algorithms = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN
}

# ---------------------------------------------------------
# Training configurations
# ---------------------------------------------------------
TOTAL_TIMESTEPS = 300_000
SAVE_DIR = "./models_compare"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------
# Training loop for all algorithms
# ---------------------------------------------------------
for name, Algo in algorithms.items():
    print(f"\n===============================================")
    print(f"Starting training for {name}...")
    print(f"===============================================\n")

    # Recreate fresh environment for each model
    env = DummyVecEnv([make_env])

    model = Algo(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=f"./tensorboard_{name.lower()}/"
    )

    # Logger setup
    new_logger = configure(f"./tensorboard_logs_{name}/", ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"./checkpoints_{name}/",
        name_prefix=f"{name.lower()}_cloud_model"
    )

    # Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback])

    # Save model
    model.save(f"{SAVE_DIR}/{name.lower()}_cloud_model_final")
    print(f"{name} training completed and model saved at {SAVE_DIR}/{name.lower()}_cloud_model_final.zip")

print("\nAll models trained successfully!")
