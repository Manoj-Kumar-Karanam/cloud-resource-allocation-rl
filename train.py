# =========================================================
# train.py
# =========================================================
# Deep Reinforcement Learning training script for
# Cloud Resource Allocation using PPO + custom reward
# =========================================================

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from cloud_env import CloudEnv
import os

def make_env():
    # Adjust PM and VM counts based on your compute power
    return CloudEnv(n_pms=8, n_vms=20, max_steps=200)

# 1. Create environment
env = DummyVecEnv([make_env])

# 2. Configure PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.2,
    tensorboard_log="./tensorboard_cloud/"
)

# 3. Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path="./checkpoints/",
    name_prefix="ppo_cloud_model"
)

eval_env = DummyVecEnv([make_env])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=10_000,
    deterministic=True,
    render=False
)

# 4. Logger for TensorBoard
new_logger = configure("./tensorboard_cloud_logs/", ["stdout", "tensorboard"])
model.set_logger(new_logger)

# 5. Train model
TOTAL_TIMESTEPS = 300_000
print(f"Starting PPO training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback])

# 6. Save model
os.makedirs("./models", exist_ok=True)
model.save("./models/ppo_cloud_model_final")
print("Training completed. Model saved successfully.")
