# =========================================================
# evaluate.py
# =========================================================
# Evaluate the trained PPO model on the CloudEnv
# =========================================================

from stable_baselines3 import PPO
from cloud_env import CloudEnv
import matplotlib.pyplot as plt
import numpy as np

# 1. Load environment and trained model
env = CloudEnv(n_pms=8, n_vms=20, max_steps=200)
model = PPO.load("./checkpoints/ppo_cloud_model_20000_steps")


# 2. Run evaluation episodes
episodes = 10
episode_rewards = []
energy_costs = []
avg_utils = []
migrations = []
qos_costs = []

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    total_energy = 0
    total_qos = 0
    total_migs = 0
    total_util = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)  # Gymnasium API
        done = terminated or truncated  # Combine both for compatibility
        total_reward += reward
        total_energy += info["energy_cost"]
        total_qos += info["qos_cost"]
        total_migs += info["migrations"]
        total_util += info["avg_util"]
        steps += 1
        env.render()


    episode_rewards.append(total_reward)
    energy_costs.append(total_energy / steps)
    qos_costs.append(total_qos / steps)
    migrations.append(total_migs)
    avg_utils.append(total_util / steps)

    print(f"Episode {ep+1}: Reward={total_reward:.2f}, "
          f"AvgUtil={np.mean(avg_utils):.2f}, Energy={np.mean(energy_costs):.2f}")

# 3. Visualization
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total Reward per Episode")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(avg_utils, label="Avg Utilization", color='orange')
plt.xlabel("Episode")
plt.ylabel("Utilization")
plt.title("Average Resource Utilization")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(energy_costs, label="Energy Cost", color='green')
plt.xlabel("Episode")
plt.ylabel("Power Usage")
plt.title("Energy Consumption per Episode")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(qos_costs, label="QoS Cost", color='red')
plt.xlabel("Episode")
plt.ylabel("QoS Violations")
plt.title("QoS Penalty per Episode")
plt.legend()

plt.tight_layout()
plt.show()

print("Evaluation completed successfully.")
