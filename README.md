# Cloud Resource Allocation with Reinforcement Learning

## Description

This repository contains a Reinforcement Learning (RL) based approach for optimizing cloud resource allocation. It leverages the power of RL to intelligently manage and distribute resources in a cloud environment, aiming to improve efficiency, reduce costs, and enhance overall performance.

## Key Features & Benefits

*   **Dynamic Resource Allocation:** Adapts to changing workload demands in real-time.
*   **Optimized Resource Utilization:** Maximizes the usage of available cloud resources.
*   **Cost Reduction:** Minimizes operational costs by efficiently allocating resources.
*   **Scalability:** Designed to handle a large number of virtual machines and physical machines.
*   **Customizable Environment:**  `cloud_env.py` allows for configuration of various cloud parameters such as number of VMs and PMs

## Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.7 or higher.
*   **pip:** Python package installer.
*   **Libraries:**
    *   `gymnasium`
    *   `numpy`
    *   `stable-baselines3`
    *   `matplotlib`
    *   `tensorboard`

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Manoj-Kumar-Karanam/cloud-resource-allocation-rl.git
    cd cloud-resource-allocation-rl
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```


3.  **Install the required packages manually if `requirements.txt` does not exist:**
    ```bash
    pip install gymnasium numpy stable-baselines3 matplotlib tensorboard
    ```

## Usage Examples & API Documentation

### Training the RL Agent

To train the Reinforcement Learning agent, use the `train.py` script:

```bash
python train.py
```

This script trains a PPO model and saves checkpoints to the `checkpoints/` directory, eval logs to the `eval_logs/` directory, and tensorboard logs to `tensorboard_cloud_logs/`.

### Evaluating the Trained Model

To evaluate the performance of the trained model, use the `evaluate.py` script:

```bash
python evaluate.py
```

This script loads a trained PPO model from the `checkpoints/` directory and evaluates it on the `CloudEnv`. The evaluation results, including a plot of the cumulative rewards over time, will be displayed.

### cloud_env.py API

The `cloud_env.py` file defines the `CloudEnv` environment, which is a custom Gymnasium environment for cloud resource allocation.  It defines `PM` and `VM` classes which manage the state of Physical Machines and Virtual Machines.

```python
#Example from cloud_env.py
class PM:
    def __init__(self, cpu_cap=100.0, mem_cap=100.0):
        self.cpu_cap = cpu_cap
        self.mem_cap = mem_cap
        self.cpu_used = 0.0
        self.mem_used = 0.0
        self.power_on = True
        self.idle_since = 0  # steps idle

    def utilization(self):
        return self.cpu_used / max(1.0, self.cpu_cap)
```

## Configuration Options

The following configuration options can be adjusted:

*   **`n_pms` (Number of Physical Machines):**  This parameter can be set inside the `CloudEnv` instantiation in `train.py` and `evaluate.py` files, impacting the scale of the simulation.
*   **`n_vms` (Number of Virtual Machines):** This parameter can be set inside the `CloudEnv` instantiation in `train.py` and `evaluate.py` files, impacting the scale of the simulation.
*   **`max_steps` (Maximum steps per episode):**  This parameter can be set inside the `CloudEnv` instantiation in `train.py` and `evaluate.py` files, impacting the simulation duration.
*   **PPO Hyperparameters:** The hyperparameters for the PPO algorithm, such as the learning rate, number of steps, and batch size, can be configured in the `train.py` script.
*   **Reward Function:** The reward function can be modified in the `cloud_env.py` file to customize the training objective.
*   **PM CPU and Memory capacity:** The `cpu_cap` and `mem_cap` parameters of the `PM` class in `cloud_env.py` can be modified to simulate different PM configurations.

## Contributing Guidelines

We welcome contributions to improve this project! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request with a clear explanation of your changes.

## License Information

This project does not specify a license. All rights are reserved unless otherwise specified.

## Acknowledgments

We would like to thank the developers of `stable-baselines3`, `gymnasium` and `numpy` for providing the core tools and libraries that made this project possible.
