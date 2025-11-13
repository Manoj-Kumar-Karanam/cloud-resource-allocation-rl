# =========================================================
# evaluate_compare.py
# =========================================================
# Evaluate PPO, A2C, DQN and generate per-algorithm plots
# =========================================================

from stable_baselines3 import PPO, A2C, DQN
from cloud_env import CloudEnv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

# Create results directory
os.makedirs("algo_plots", exist_ok=True)

# Load trained models
models = {
    "PPO": PPO.load("./models_compare/ppo_cloud_model_final"),
    "A2C": A2C.load("./models_compare/a2c_cloud_model_final"),
    "DQN": DQN.load("./models_compare/dqn_cloud_model_final")
}

episodes = 10

def evaluate_and_plot(name, model):

    print(f"\nEvaluating {name}...")

    env = CloudEnv(n_pms=8, n_vms=20, max_steps=200)

    episode_rewards = []
    energy_costs = []
    avg_utils = []
    qos_costs = []

    pbar = tqdm(total=episodes, desc=f"Evaluating {name}", unit="episode")

    for ep in range(episodes):

        obs, info = env.reset()
        done = False
        total_reward = 0
        total_energy = 0
        total_qos = 0
        total_util = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_energy += info["energy_cost"]
            total_qos += info["qos_cost"]
            total_util += info["avg_util"]
            steps += 1

        episode_rewards.append(total_reward)
        energy_costs.append(total_energy / steps)
        avg_utils.append(total_util / steps)
        qos_costs.append(total_qos / steps)

        pbar.update(1)

    pbar.close()

    # ------------------------------------------------------
    # Generate a single figure with 4 plots
    # ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # 1. Reward Plot
    axes[0, 0].plot(episode_rewards, label="Episode Reward")
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    # 2. Utilization Plot
    axes[0, 1].plot(avg_utils, label="Avg Utilization", color='orange')
    axes[0, 1].set_title("Average Resource Utilization")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Utilization")
    axes[0, 1].legend()

    # 3. Energy Plot
    axes[1, 0].plot(energy_costs, label="Energy Cost", color='green')
    axes[1, 0].set_title("Energy Consumption per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Power Usage")
    axes[1, 0].legend()

    # 4. QoS Penalty Plot
    axes[1, 1].plot(qos_costs, label="QoS Cost", color='red')
    axes[1, 1].set_title("QoS Penalty per Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("QoS Violations")
    axes[1, 1].legend()

    plt.suptitle(f"{name} Performance on CloudEnv", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"algo_plots/{name.lower()}_results.png")
    plt.show()

    print(f"{name} plot saved as algo_plots/{name.lower()}_results.png")


# ------------------------------------------------------------
# Evaluate all algorithms
# ------------------------------------------------------------
for algo_name, model in models.items():
    evaluate_and_plot(algo_name, model)

print("\nAll algorithm plots generated successfully.")
