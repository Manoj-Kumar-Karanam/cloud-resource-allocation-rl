# cloud_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

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

class VM:
    def __init__(self, vm_id, cpu_req=10.0, mem_req=10.0):
        self.vm_id = vm_id
        self.cpu_req = cpu_req
        self.mem_req = mem_req
        self.host = None

class CloudEnv(gym.Env):
    """
    Gym-like environment for simplified cloud resource allocation.
    Action space compresses to: {NoOp, Migrate(vm, target_pm), PowerManage(pm)}.
    To keep action discrete, we enumerate candidate actions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_pms=6, n_vms=12, max_steps=200, top_k_candidates=3):
        super().__init__()
        self.n_pms = n_pms
        self.n_vms = n_vms
        self.max_steps = max_steps
        self.top_k = top_k_candidates

        # create PMs and VMs
        self.pms = [PM(cpu_cap=100.0, mem_cap=100.0) for _ in range(n_pms)]
        self.vms = [VM(vm_id=i, cpu_req=random.uniform(5, 30), mem_req=random.uniform(5,30)) for i in range(n_vms)]

        # observation: for each PM -> [util, powered_on, idle_norm], plus global stats
        obs_len = self.n_pms * 3 + 3  # add 3 summary features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # discrete actions by enumerating candidates:
        # action 0: NoOp
        # actions 1..M: Migrate top-k overloaded VMs to top-k target PMs => we define candidate list at each step
        # next chunk: PowerManage(pm_id) for each pm -> map to discrete index
        # We'll build action mapping dynamically in step() and require agent to choose valid action index.
        # To allow learning frameworks to have fixed size action space, we define a max_actions cap.
        self.max_actions = 1 + self.top_k * self.top_k + self.n_pms
        self.action_space = spaces.Discrete(self.max_actions)

        self.current_step = 0

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        Compatible with Gymnasium API (returns obs, info).
        """
        super().reset(seed=seed)

        # reset PMs and VMs
        for pm in self.pms:
            pm.cpu_used = 0.0
            pm.mem_used = 0.0
            pm.power_on = True
            pm.idle_since = 0

        for vm in self.vms:
            vm.cpu_req = random.uniform(5, 30)
            vm.mem_req = random.uniform(5, 30)
            host = random.randrange(self.n_pms)
            vm.host = host
            self.pms[host].cpu_used += vm.cpu_req
            self.pms[host].mem_used += vm.mem_req

        self.current_step = 0
        self._build_action_candidates()

        obs = self._get_obs()
        info = {}  # optional extra info (can add metrics later)
        return obs, info


    def _get_obs(self):
        pm_feats = []
        overloaded = 0
        for pm in self.pms:
            util = pm.utilization()
            pm_feats.extend([util, 1.0 if pm.power_on else 0.0, min(1.0, pm.idle_since / 50.0)])
            if util > 1.0:
                overloaded += 1
        # summary: overloaded fraction, powered on fraction, avg util
        powered_on_frac = sum(1 for pm in self.pms if pm.power_on) / self.n_pms
        avg_util = np.mean([pm.utilization() for pm in self.pms])
        overloaded_frac = overloaded / self.n_pms
        obs = np.array(pm_feats + [powered_on_frac, avg_util, overloaded_frac], dtype=np.float32)
        return obs

    def _power_of_pm(self, pm: PM):
        # simple linear model: power = base + slope * util
        base = 50.0
        slope = 50.0
        if not pm.power_on:
            return 0.0
        else:
            return base + slope * pm.utilization()

    def _total_power(self):
        return sum(self._power_of_pm(pm) for pm in self.pms)

    def _build_action_candidates(self):
        # determine top-k overloaded VMs (by their host's util)
        vm_scores = []
        for vm in self.vms:
            host_util = self.pms[vm.host].utilization()
            vm_scores.append((host_util, vm))
        vm_scores.sort(reverse=True, key=lambda x: x[0])
        self.migrate_candidates = [vm for _, vm in vm_scores[:self.top_k]]

        # target PMs: powered on pm with lower util than source, plus powered off ones
        # For candidate enumeration, we select top_k PMs with lowest util
        pm_by_util = sorted(self.pms, key=lambda p: p.utilization())
        self.target_candidates = pm_by_util[:self.top_k]

        # build mapping from discrete action index to (type, args)
        self.action_map = {}
        idx = 0
        self.action_map[idx] = ("NoOp", None)  # idx 0
        idx += 1
        # migrate actions
        for vm in self.migrate_candidates:
            for target in self.target_candidates:
                if target is self.pms[vm.host]:
                    continue
                if idx >= self.max_actions: break
                self.action_map[idx] = ("Migrate", (vm.vm_id, self.pms.index(target)))
                idx += 1
        # power manage actions (toggle)
        for pm_id in range(self.n_pms):
            if idx >= self.max_actions: break
            self.action_map[idx] = ("PowerManage", (pm_id,))
            idx += 1
        # fill remaining indices as NoOp to keep action_space consistent
        while idx < self.max_actions:
            self.action_map[idx] = ("NoOp", None)
            idx += 1

    def step(self, action):
        """
        Executes one environment step based on the RL agent's chosen action.
        Applies Migrate, PowerManage, or NoOp actions, updates VM/PM states,
        and computes the multi-objective reward function.
        """
        if isinstance(action, (np.ndarray, list)):
            action = action.item()

        assert self.action_space.contains(action)
        self.current_step += 1

        # Rebuild action candidates for this step (since system state may change)
        self._build_action_candidates()
        act_type, args = self.action_map.get(action, ("NoOp", None))

        migration_count = 0
        migration_cost = 0.0

        # ------------------------------
        # 1. Execute selected action
        # ------------------------------
        if act_type == "Migrate" and args is not None:
            vm_id, target_pm = args
            vm = self.vms[vm_id]
            src = vm.host
            if src != target_pm and self.pms[target_pm].power_on:
                # Perform migration: update host resource allocations
                self.pms[src].cpu_used -= vm.cpu_req
                self.pms[src].mem_used -= vm.mem_req
                self.pms[target_pm].cpu_used += vm.cpu_req
                self.pms[target_pm].mem_used += vm.mem_req
                vm.host = target_pm
                migration_count = 1
                migration_cost = vm.mem_req * 0.1  # proportional to memory size

        elif act_type == "PowerManage" and args is not None:
            pm_id = args[0]
            pm = self.pms[pm_id]
            # If idle, power off; else if off but has assigned VM(s), power on
            if pm.cpu_used < 1e-6 and pm.power_on:
                pm.power_on = False
                pm.idle_since = 0
            elif (not pm.power_on) and any(vm.host == pm_id for vm in self.vms):
                pm.power_on = True

        # ------------------------------
        # 2. Simulate next time step
        # ------------------------------
        for vm in self.vms:
            # Add small fluctuation to VM CPU demand
            vm.cpu_req = max(1.0, vm.cpu_req + random.uniform(-2.0, 2.0))

        # Recompute PM resource utilization from VM allocations
        for pm in self.pms:
            pm.cpu_used = 0.0
            pm.mem_used = 0.0
        for vm in self.vms:
            host = vm.host
            self.pms[host].cpu_used += vm.cpu_req
            self.pms[host].mem_used += vm.mem_req

        # Update idle timers for PMs
        for pm in self.pms:
            if pm.cpu_used < 1e-6 and pm.power_on:
                pm.idle_since += 1
            else:
                pm.idle_since = 0

        # ------------------------------
        # 3. Compute system metrics
        # ------------------------------
        total_power = self._total_power()
        overloaded_hosts = sum(1 for pm in self.pms if pm.utilization() > 1.0)

        # ------------------------------
        # 4. Multi-objective Reward
        # ------------------------------
        # λ weighting coefficients
        lambda1, lambda2, lambda3, lambda4 = 0.6, 0.2, 0.15, 0.05

        # (a) Resource Utilization Reward
        avg_utilization = np.mean([pm.utilization() for pm in self.pms if pm.power_on])

        # (b) Energy Cost (power consumption)
        energy_cost = total_power

        # (c) QoS Cost (SLA violations due to overload)
        qos_cost = overloaded_hosts * 50.0

        # (d) Migration Cost (network/memory overhead)
        mig_cost = migration_cost

        # Combine into overall reward
        reward = (
            lambda1 * avg_utilization
            - lambda2 * energy_cost
            - lambda3 * qos_cost
            - lambda4 * mig_cost
        )

        # Optional normalization (recommended for stability)
        reward = float(np.clip(reward, -1000, 1000))

        # ------------------------------
        # 5. End of step processing
        # ------------------------------
        done = self.current_step >= self.max_steps
        obs = self._get_obs()
        info = dict(
            total_power=total_power,
            overloaded=overloaded_hosts,
            migrations=migration_count,
            avg_util=avg_utilization,
            energy_cost=energy_cost,
            qos_cost=qos_cost,
            mig_cost=mig_cost
        )

        # Rebuild candidates for next step
        self._build_action_candidates()
        return obs, reward, done, False, info
    def render(self):
        util = [round(pm.utilization(), 2) for pm in self.pms]
        print(f"Step {self.current_step} | PM utils: {util}")


