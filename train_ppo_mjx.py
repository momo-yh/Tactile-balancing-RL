import argparse
import mujoco
import mujoco.mjx
import jax
import jax.numpy as jnp
import xml.etree.ElementTree as ET
import numpy as np
import time
from datetime import datetime
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.distributions import Normal
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import os
import sys
import json
import settings as settings_mod
from reward import (
    ALIVE_BONUS,
    FALL_PENALTY,
    SENSOR_MATCH_BONUS,
    SENSOR_MATCH_TOLERANCE,
)
from settings import (
    STACK_SIZE, OBS_DIM, STATE_DIM, ACTION_DIM, HIDDEN_DIM,
    ACTION_FORCE_MAGNITUDE, PERTURB_FORCE_SCALE, PERTURB_APPLY_STEPS,
    PERTURB_START_DELAY, PERTURB_RESAMPLE_EPISODES, ACTION_START_DELAY,
    BOUNDARY_LIMIT, TILT_LIMIT_RADIANS, USE_OBS_NORMALIZATION,
    PERTURB_CURRICULUM_RANGES, CURRICULUM_STAGE_EPISODES,
)
from features import (
    build_observation,
    compute_tilt_angle,
    SENSOR_NAMES,
    MID_ROW_INDICES,
    ANGULAR_VEL_DIRECTION_THRESHOLD,
)

if __name__ == "__main__":
    # Ensure the module is discoverable as 'train_box_ppo' even when executed as a script.
    sys.modules.setdefault("train_box_ppo", sys.modules[__name__])

# Device selection (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PPO Core Components ---

class StateStacker:
    """Helper class to stack recent observations."""
    def __init__(self, obs_dim, stack_size):
        self.obs_dim = obs_dim
        self.stack_size = stack_size
        self.buffer = deque(maxlen=stack_size)
        self.reset()

    def reset(self, initial_obs=None):
        """Reset and fill the buffer."""
        if initial_obs is None:
            initial_obs = np.zeros(self.obs_dim)
        
        self.buffer.clear()
        for _ in range(self.stack_size):
            self.buffer.append(initial_obs)

    def push(self, obs):
        """Add a new observation."""
        self.buffer.append(obs)

    def get_state(self):
        """Get the stacked state vector."""
        return np.concatenate(self.buffer, axis=0)


class BatchedStateStacker:
    """Stacks observations for a batch of environments."""

    def __init__(self, batch_size, obs_dim, stack_size):
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.stack_size = stack_size
        self.buffer = np.zeros((stack_size, batch_size, obs_dim), dtype=np.float32)

    def reset(self, initial_obs=None, env_indices=None):
        """Reset stacked observations for all envs or a subset."""
        if initial_obs is None:
            initial_obs = np.zeros((self.batch_size, self.obs_dim), dtype=np.float32)
        if env_indices is None:
            if initial_obs.shape != (self.batch_size, self.obs_dim):
                raise ValueError("initial_obs must have shape (batch_size, obs_dim)")
            self.buffer[...] = initial_obs[np.newaxis, ...]
        else:
            env_indices = np.asarray(env_indices, dtype=np.int32)
            if initial_obs.shape == (self.batch_size, self.obs_dim):
                subset = initial_obs[env_indices]
            elif initial_obs.shape == (env_indices.size, self.obs_dim):
                subset = initial_obs
            else:
                raise ValueError("initial_obs shape mismatch for partial reset")
            self.buffer[:, env_indices, :] = subset[np.newaxis, ...]

    def push(self, obs):
        if obs.shape != (self.batch_size, self.obs_dim):
            raise ValueError("obs must have shape (batch_size, obs_dim)")
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1] = obs

    def get_state(self):
        return self.buffer.reshape(self.batch_size, -1)


def _replicate_mjx_data(single_data, batch_size):
    def _expand(arr):
        return jnp.repeat(jnp.expand_dims(jnp.asarray(arr), axis=0), batch_size, axis=0)

    return jax.tree_util.tree_map(_expand, single_data)


def _load_sanitized_model(model_path):
    with open(model_path, "r", encoding="utf-8") as fh:
        xml_text = fh.read()
    root = ET.fromstring(xml_text)
    for actuator in root.findall("actuator"):
        root.remove(actuator)
    sanitized_xml = ET.tostring(root, encoding="unicode")
    mj_model = mujoco.MjModel.from_xml_string(sanitized_xml)
    return mj_model, sanitized_xml


def _update_env_slices(batch_tree, env_indices, *single_trees):
    def _update_leaf(batch_leaf, *single_leaf_list):
        batch_np = np.asarray(batch_leaf).copy()
        for idx, single_leaf in zip(env_indices, single_leaf_list):
            batch_np[idx] = single_leaf
        return batch_np

    return jax.tree_util.tree_map(_update_leaf, batch_tree, *single_trees)


def _compute_tilt_angle_batch(quat_batch):
    w = quat_batch[:, 0]
    x = quat_batch[:, 1]
    y = quat_batch[:, 2]
    z = quat_batch[:, 3]
    z_axis_x = 2.0 * (x * z - w * y)
    z_axis_z = 1.0 - 2.0 * (x ** 2 + y ** 2)
    return np.arctan2(z_axis_x, z_axis_z)


def _compute_reward_batch(contact_batch, tilt_angles, tilt_rates, action_forces, out_of_bounds_flags, tilt_limit):
    contact_batch = np.nan_to_num(contact_batch, nan=0.0, posinf=0.0, neginf=0.0)
    tilt_abs = np.abs(tilt_angles)
    alive_mask = (tilt_abs <= tilt_limit) & (~out_of_bounds_flags)

    rewards = np.empty_like(tilt_angles, dtype=np.float32)
    rewards[:] = -FALL_PENALTY

    if np.any(alive_mask):
        alive_idx = np.where(alive_mask)[0]
        base_reward = np.full(alive_idx.shape, ALIVE_BONUS, dtype=np.float32)

        if SENSOR_MATCH_BONUS > 0.0:
            sensors = contact_batch[alive_idx]
            spread = np.max(sensors, axis=1) - np.min(sensors, axis=1)
            safe_tol = np.maximum(SENSOR_MATCH_TOLERANCE, 1e-8)
            match_score = np.clip(1.0 - spread / safe_tol, 0.0, 1.0)
            base_reward += SENSOR_MATCH_BONUS * match_score

        rewards[alive_idx] = base_reward

    return rewards.astype(np.float32), alive_mask


def _make_mjx_step(model):
    def single_step(data):
        return mujoco.mjx.step(model, data)

    @jax.jit
    def batched_step(data, xfrc_applied, ctrl):
        data = data.replace(xfrc_applied=xfrc_applied, ctrl=ctrl)
        return jax.vmap(single_step)(data)

    return batched_step


class MJXParallelEnv:
    def __init__(self, model_path, batch_size):
        self.num_envs = batch_size
        self.model, self.xml_string = _load_sanitized_model(model_path)
        self.mjx_model = mujoco.mjx.put_model(self.model)
        self._step_fn = _make_mjx_step(self.mjx_model)

        self.nbodies = self.model.nbody
        self.nu = self.model.nu
        self.box_body_id = self.model.body('box_body').id

        self.sensor_indices = np.array([
            self.model.sensor(name).adr for name in SENSOR_NAMES
        ], dtype=np.int32)
        self.middle_sensor_indices = np.array(MID_ROW_INDICES, dtype=np.int32)
        self._sensor_indices_device = jnp.asarray(self.sensor_indices)
        self._vel_threshold = float(ANGULAR_VEL_DIRECTION_THRESHOLD)

        cpu_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, cpu_data)
        self._cpu_reset_data = cpu_data
        single_mjx_data = mujoco.mjx.put_data(self.model, cpu_data)

        self.mjx_data = _replicate_mjx_data(single_mjx_data, batch_size)
        self._default_ctrl = np.zeros((batch_size, self.nu), dtype=np.float32)

        self.reset()

    def _reset_single(self):
        mujoco.mj_resetData(self.model, self._cpu_reset_data)
        angle = np.random.uniform(-0.1, 0.1)
        axis = np.random.normal(size=3)
        axis /= (np.linalg.norm(axis) + 1e-12)
        qw = np.cos(angle * 0.5)
        qxyz = axis * np.sin(angle * 0.5)
        self._cpu_reset_data.qpos[3:7] = [float(qw), float(qxyz[0]), float(qxyz[1]), float(qxyz[2])]
        mujoco.mj_forward(self.model, self._cpu_reset_data)
        return mujoco.mjx.put_data(self.model, self._cpu_reset_data)

    def _sensor_grid(self):
        sensor_vals = jnp.take(self.mjx_data.sensordata, self._sensor_indices_device, axis=1)
        sensors = np.asarray(sensor_vals, dtype=np.float32)
        if sensors.ndim == 3 and sensors.shape[-1] == 1:
            sensors = np.squeeze(sensors, axis=-1)
        return sensors

    def _mid_row_contacts(self):
        sensors = self._sensor_grid()
        return sensors[:, self.middle_sensor_indices]

    def _angular_velocity_direction_batch(self):
        cvel = np.asarray(self.mjx_data.cvel[:, self.box_body_id, 1], dtype=np.float32)
        direction = np.zeros_like(cvel)
        direction[cvel > self._vel_threshold] = 1.0
        direction[cvel < -self._vel_threshold] = -1.0
        return direction

    def _compute_observation_components(self):
        mid_row = self._mid_row_contacts().astype(np.float32)
        tilt_proxy = mid_row[:, 0] - mid_row[:, -1]
        vel_direction = self._angular_velocity_direction_batch().astype(np.float32)
        obs = np.stack([tilt_proxy, vel_direction], axis=1)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, mid_row

    def reset(self, env_indices=None):
        if env_indices is None:
            env_indices = np.arange(self.num_envs)
        else:
            env_indices = np.asarray(env_indices, dtype=np.int32)
        current = jax.tree_util.tree_map(np.asarray, jax.device_get(self.mjx_data))
        singles = [jax.tree_util.tree_map(np.asarray, self._reset_single()) for _ in env_indices]
        updated = _update_env_slices(current, env_indices, *singles)
        self.mjx_data = jax.tree_util.tree_map(jnp.asarray, updated)
        obs = self.get_observation()
        return obs

    def get_observation(self):
        obs, _ = self._compute_observation_components()
        return obs

    def step(self, action_forces, perturb_forces):
        xfrc = np.zeros((self.num_envs, self.nbodies, 6), dtype=np.float32)
        xfrc[:, self.box_body_id, 0] = action_forces
        if perturb_forces is not None:
            xfrc[:, self.box_body_id, :] += perturb_forces
        xfrc = jnp.asarray(xfrc)
        ctrl = jnp.asarray(self._default_ctrl)
        self.mjx_data = self._step_fn(self.mjx_data, xfrc, ctrl)
        obs_features, mid_contacts = self._compute_observation_components()
        quat = np.asarray(self.mjx_data.xquat[:, self.box_body_id], dtype=np.float32)
        tilt_angles = _compute_tilt_angle_batch(quat)
        tilt_rates = np.asarray(self.mjx_data.cvel[:, self.box_body_id, 1], dtype=np.float32)
        xpos = np.asarray(self.mjx_data.xpos[:, self.box_body_id], dtype=np.float32)
        return obs_features, mid_contacts, tilt_angles, tilt_rates, xpos


def _sample_startup_perturbation(force_range=None):
    """Sample a startup perturbation force along x using signed intervals.

    force_range: tuple (min_abs, max_abs) describing absolute magnitudes. When
    omitted, +/- PERTURB_FORCE_SCALE is used. Sampling respects the minimum
    magnitude while extending the interval symmetrically into the negative
    direction as requested.
    """
    if force_range is None:
        min_abs = 0.0
        max_abs = float(PERTURB_FORCE_SCALE)
    else:
        min_raw, max_raw = force_range
        min_abs = abs(float(min_raw))
        max_abs = abs(float(max_raw))

    if max_abs < min_abs:
        min_abs, max_abs = max_abs, min_abs

    if max_abs == 0.0:
        force_value = 0.0
    elif min_abs <= 0.0:
        low = -max_abs
        high = max_abs
        force_value = np.random.uniform(low, high) if high > low else float(high)
    else:
        magnitude = np.random.uniform(min_abs, max_abs) if max_abs > min_abs else float(min_abs)
        sign = -1.0 if np.random.rand() < 0.5 else 1.0
        force_value = sign * magnitude

    force = np.zeros(6, dtype=np.float32)
    force[0] = float(force_value)
    return force


class RunningMeanStd:
    """Running mean/std for observation normalization (numerically stable)."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        # Handle single sample (1-D) and batches consistently
        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1.0
        else:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        denom = np.sqrt(np.maximum(self.var, 1e-6))
        return ((x - self.mean) / (denom + 1e-8)).astype(np.float32)


def np_scalar(value):
    """Convert a NumPy scalar (or scalar-like) value to a Python number without using float()."""
    return np.asarray(value).item()

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
LOG_RATIO_CLIP = 10.0


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO using a Gaussian policy with tanh-squash.

    Outputs a mean vector from the actor network and keeps a learnable
    log_std parameter per-action. Actions are sampled from Normal(mean, std)
    and then squashed with tanh to lie in [-1, 1]. Log-probabilities account
    for the tanh change-of-variables.
    """
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim

        # actor outputs mean for Gaussian
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # learnable log std (initialized near 0)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        # ensure tensor on correct device
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        mean = self.actor(state)
        value = self.critic(state)

        log_std_param = torch.nan_to_num(
            self.log_std,
            nan=LOG_STD_MIN,
            posinf=LOG_STD_MAX,
            neginf=LOG_STD_MIN,
        )
        log_std = torch.clamp(log_std_param, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            # reparameterized sample for stable training
            pre_tanh = dist.rsample()
            action_t = torch.tanh(pre_tanh)

            # log_prob correction for tanh squashing
            # sum over action dims
            log_prob = dist.log_prob(pre_tanh) - torch.log(1 - action_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # return numpy action for env, plus tensors for learning
            return action_t.cpu().numpy(), log_prob, entropy, value
        else:
            # `action` is expected to be the squashed action in [-1,1]
            if not torch.is_tensor(action):
                action = torch.as_tensor(action, dtype=torch.float32, device=device)
            else:
                action = action.to(device).float()

            # ensure batch dim
            if action.dim() == 1:
                action = action.unsqueeze(0)

            # invert tanh to get pre-squash value
            # atanh(x) = 0.5 * (log1p(x) - log1p(-x))
            eps = 1e-6
            clipped = action.clamp(-1 + eps, 1 - eps)
            pre_tanh = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))

            log_prob = dist.log_prob(pre_tanh) - torch.log(1 - clipped.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return action, log_prob, entropy, value

class RolloutBuffer:
    """Storage for PPO rollout data across batched environments."""

    def __init__(self, rollout_length, num_envs, obs_dim, action_dim, gamma, gae_lambda):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((rollout_length, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_length, num_envs), dtype=np.float32)

        self.step = 0

    def store(self, obs_batch, action_batch, reward_batch, done_batch, log_prob_batch, value_batch):
        if self.step >= self.rollout_length:
            print(
                f"[RolloutBuffer] Warning: buffer index {self.step} >= rollout_length {self.rollout_length}. Resetting step to 0 and overwriting."
            )
            self.step = 0

        self.observations[self.step] = obs_batch
        self.actions[self.step] = action_batch
        self.rewards[self.step] = reward_batch
        self.dones[self.step] = done_batch
        self.log_probs[self.step] = log_prob_batch
        self.values[self.step] = value_batch
        self.step += 1

    def compute_advantages(self, last_value, last_done):
        last_value = np.asarray(last_value, dtype=np.float32)
        last_done = np.asarray(last_done, dtype=np.float32)
        gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values
        self.step = 0

    def get_minibatches(self, batch_size):
        total_samples = self.rollout_length * self.num_envs
        indices = np.random.permutation(total_samples)

        flat_obs = self.observations.reshape(total_samples, -1)
        flat_actions = self.actions.reshape(total_samples, -1)
        flat_log_probs = self.log_probs.reshape(total_samples)
        flat_advantages = self.advantages.reshape(total_samples)
        flat_returns = self.returns.reshape(total_samples)

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]
            obs_b = torch.tensor(flat_obs[mb_indices], dtype=torch.float32, device=device)
            actions_b = torch.tensor(flat_actions[mb_indices], dtype=torch.float32, device=device)

            yield (
                obs_b,
                actions_b,
                torch.tensor(flat_log_probs[mb_indices], dtype=torch.float32, device=device),
                torch.tensor(flat_advantages[mb_indices], dtype=torch.float32, device=device),
                torch.tensor(flat_returns[mb_indices], dtype=torch.float32, device=device),
            )

class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        rollout_length=2048,
        batch_size=64,
        ppo_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        num_envs=1,
    ):
        self.network = ActorCritic(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = None

        self.buffer = RolloutBuffer(rollout_length, num_envs, state_dim, action_dim, gamma, gae_lambda)

        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.step_count = 0
        self.rollout_length = rollout_length
        self.update_count = 0
        self.num_envs = num_envs

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state)
        if torch.is_tensor(action):
            action_np = action.cpu().numpy()
        else:
            action_np = np.asarray(action)
        return action_np, log_prob, value

    def store_transition(self, obs_batch, action_batch, reward_batch, done_batch, log_prob_batch, value_batch):
        obs_arr = np.asarray(obs_batch, dtype=np.float32)
        action_arr = np.asarray(action_batch, dtype=np.float32)
        reward_arr = np.asarray(reward_batch, dtype=np.float32)
        done_arr = np.asarray(done_batch, dtype=np.float32)
        log_prob_arr = np.asarray(log_prob_batch, dtype=np.float32)
        value_arr = np.asarray(value_batch, dtype=np.float32)

        if action_arr.ndim == 1:
            action_arr = action_arr[:, np.newaxis]
        if log_prob_arr.ndim > 1:
            log_prob_arr = log_prob_arr.reshape(-1)
        if value_arr.ndim == 2 and value_arr.shape[1] == 1:
            value_arr = value_arr[:, 0]

        if not np.isfinite(obs_arr).all() or not np.isfinite(action_arr).all() or not np.isfinite(log_prob_arr).all() or not np.isfinite(value_arr).all():
            print("[WARN] Skipping transition batch with non-finite data.")
            return False

        obs_sanitized = np.nan_to_num(obs_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        action_sanitized = np.nan_to_num(action_arr, nan=0.0, posinf=1.0, neginf=-1.0)

        self.buffer.store(obs_sanitized, action_sanitized, reward_arr, done_arr, log_prob_arr, value_arr)
        self.step_count += 1
        return self.step_count % self.rollout_length == 0

    def update(self, last_state, last_done):
        """Perform PPO update."""
        if not torch.is_tensor(last_state):
            last_state_tensor = torch.as_tensor(last_state, dtype=torch.float32, device=device)
        else:
            last_state_tensor = last_state.to(device)

        with torch.no_grad():
            last_value = self.network.get_value(last_state_tensor)
        last_value_np = last_value.squeeze(-1).cpu().numpy()
        self.buffer.compute_advantages(last_value_np, last_done)

        advantages = self.buffer.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.reshape(self.buffer.advantages.shape)
        
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_minibatches(self.batch_size):
                b_obs, b_actions, b_old_log_probs, b_advantages, b_returns = batch

                if not torch.isfinite(b_obs).all() or not torch.isfinite(b_actions).all() or \
                   not torch.isfinite(b_old_log_probs).all() or not torch.isfinite(b_advantages).all() or \
                   not torch.isfinite(b_returns).all():
                    print("[WARN] Skipping PPO minibatch due to non-finite rollout data.")
                    continue

                # For discrete actions, b_actions is a long tensor of indices; ActorCritic will accept that
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(b_obs, b_actions)
                new_values = new_values.squeeze()

                if (not torch.isfinite(new_log_probs).all()) or (not torch.isfinite(new_values).all()):
                    print("[WARN] Skipping PPO minibatch due to non-finite policy outputs.")
                    continue
                if entropy is not None and (not torch.isfinite(entropy).all()):
                    print("[WARN] Entropy contained non-finite values; using zero entropy contribution.")
                    entropy = torch.zeros_like(b_advantages)

                # new_log_probs and b_old_log_probs are both shaped (batch,)
                log_ratio = new_log_probs - b_old_log_probs
                if not torch.isfinite(log_ratio).all():
                    print("[WARN] Skipping PPO minibatch due to non-finite log probability difference.")
                    continue
                log_ratio = torch.clamp(log_ratio, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
                ratio = torch.exp(log_ratio)
                if not torch.isfinite(ratio).all():
                    print("[WARN] Skipping PPO minibatch due to non-finite probability ratio after clamping.")
                    continue
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                if (not torch.isfinite(surr1).all()) or (not torch.isfinite(surr2).all()):
                    print("[WARN] Skipping PPO minibatch due to non-finite surrogate losses.")
                    continue
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, b_returns)
                try:
                    entropy_loss = -entropy.mean()
                except Exception:
                    # entropy can be None or a different shape; fall back to zero
                    entropy_loss = 0.0

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    sanitized = torch.nan_to_num(
                        self.network.log_std.data,
                        nan=LOG_STD_MIN,
                        posinf=LOG_STD_MAX,
                        neginf=LOG_STD_MIN,
                    )
                    self.network.log_std.data.copy_(sanitized.clamp_(LOG_STD_MIN, LOG_STD_MAX))

        self.update_count += 1
        if self.scheduler is not None:
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < prev_lr - 1e-12:
                print(f"[LR Scheduler] Update {self.update_count}: lr reduced from {prev_lr:.6g} to {new_lr:.6g}")

# --- MuJoCo Environment Utilities ---

def get_observation(data):
    return build_observation(data)

def is_done(data):
    """Episode terminates when tilt exceeds configured threshold."""
    return abs(compute_tilt_angle(data)) > TILT_LIMIT_RADIANS

def reset_sim(model, data, state_stacker, obs_rms=None):
    """Reset the simulation."""
    mujoco.mj_resetData(model, data)
    # Give a small random initial rotation using a normalized quaternion.
    # Sample a small random axis-angle perturbation (angle in radians up to 0.1)
    angle = np.random.uniform(-0.1, 0.1)
    axis = np.random.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    qw = np.cos(angle * 0.5)
    qxyz = axis * np.sin(angle * 0.5)
    # quaternion ordering in qpos is [w, x, y, z]
    data.qpos[3:7] = [float(qw), float(qxyz[0]), float(qxyz[1]), float(qxyz[2])]
    mujoco.mj_forward(model, data)

    initial_obs = get_observation(data)
    initial_obs = np.nan_to_num(initial_obs, nan=0.0, posinf=1e6, neginf=-1e6)
    if obs_rms is not None:
        obs_rms.update(initial_obs)
        normalized = obs_rms.normalize(initial_obs)
        if not np.isfinite(normalized).all():
            print("[WARN] Non-finite normalized reset observation; using zeros.")
            normalized = np.zeros_like(normalized)
        else:
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1e6, neginf=-1e6)
        state_stacker.reset(normalized)
    else:
        state_stacker.reset(initial_obs)

    return state_stacker.get_state()

# --- Main Training Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-dir', type=str, default=None,
                        help='Optional directory containing saved model/obs_rms files to resume from.')
    parser.add_argument('--resume-tag', type=str, default='best', choices=['best', 'last'],
                        help='Which checkpoint suffix to load from resume dir (default: best).')
    args = parser.parse_args()

    NUM_ENVS = int(os.environ.get("MJX_NUM_ENVS", os.environ.get("PPO_NUM_ENVS", "32")))
    MAX_TRAINING_STEPS = 6000000
    ROLLOUT_LENGTH = 128
    BATCH_SIZE = 128
    PPO_EPOCHS = 16
    LR = 1e-4
    ENTROPY_COEF = 0.01
    LR_DECAY_FRACTION = float(os.environ.get("LR_DECAY_FRACTION", "0.75"))
    LR_DECAY_GAMMA = float(os.environ.get("LR_DECAY_GAMMA", "0.1"))
    MAX_EPISODE_STEPS = 2000

    if os.environ.get("SHORT_TEST", "0") == "1":
        MAX_TRAINING_STEPS = 20000
        ROLLOUT_LENGTH = 256
        NUM_ENVS = min(NUM_ENVS, 8)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tilt_deg = int(round(np.degrees(TILT_LIMIT_RADIANS)))
    save_root = os.path.join("pretrained", f"{ts}_TILT{tilt_deg}_RL")
    latest_dir = os.path.join("pretrained", "latest")
    os.makedirs(save_root, exist_ok=True)

    env = MJXParallelEnv("model.xml", NUM_ENVS)
    eval_model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(eval_model)
    box_body_id = data.body('box_body').id

    agent = PPOAgent(
        STATE_DIM,
        ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        rollout_length=ROLLOUT_LENGTH,
        batch_size=BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        lr=LR,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=ENTROPY_COEF,
        num_envs=NUM_ENVS,
    )

    if LR_DECAY_GAMMA < 0.999:
        LR_DECAY_FRACTION = min(max(LR_DECAY_FRACTION, 0.0), 1.0)
        approx_iterations = max(1, MAX_TRAINING_STEPS // max(1, NUM_ENVS))
        expected_updates = max(1, approx_iterations // ROLLOUT_LENGTH)
        decay_update = max(1, int(expected_updates * LR_DECAY_FRACTION))
        scheduler = MultiStepLR(agent.optimizer, milestones=[decay_update], gamma=LR_DECAY_GAMMA)
        agent.set_scheduler(scheduler)
        print(f"[LR Scheduler] MultiStepLR enabled: decay after {decay_update} updates (≈{expected_updates} total), gamma={LR_DECAY_GAMMA}")

    state_stacker = BatchedStateStacker(NUM_ENVS, OBS_DIM, STACK_SIZE)
    obs_rms = RunningMeanStd(shape=(OBS_DIM,)) if USE_OBS_NORMALIZATION else None

    if args.resume_dir:
        try:
            ckpt_suffix = args.resume_tag.lower()
            model_path = os.path.join(args.resume_dir, f'box_ppo_model_{ckpt_suffix}.pt')
            obs_path = os.path.join(args.resume_dir, f'obs_rms_{ckpt_suffix}.npz')

            state_dict = torch.load(model_path, map_location=device)
            agent.network.load_state_dict(state_dict)

            if obs_rms is not None and os.path.exists(obs_path):
                stats = np.load(obs_path)
                obs_rms.mean = stats['mean']
                obs_rms.var = stats['var']
                obs_rms.count = stats['count']
                print(f"Resumed observation stats from {obs_path}")
            elif obs_rms is None:
                if os.path.exists(obs_path):
                    print(f"Observation normalization disabled; skipping stats from {obs_path}")
            else:
                print(f"Warning: observation stats file missing at {obs_path}")

            print(f"Resumed model weights from {model_path}")
            # Record resume provenance: copy resume files into our save_root and write a resume_info file
            try:
                resume_out = os.path.join(save_root, 'resume_files')
                os.makedirs(resume_out, exist_ok=True)
                if os.path.exists(model_path):
                    shutil.copy(model_path, os.path.join(resume_out, os.path.basename(model_path)))
                if os.path.exists(obs_path):
                    shutil.copy(obs_path, os.path.join(resume_out, os.path.basename(obs_path)))

                # collect uppercase settings from settings.py into a serializable dict
                try:
                    settings_snapshot = {}
                    for name in dir(settings_mod):
                        if name.isupper():
                            val = getattr(settings_mod, name)
                            # convert numpy scalars or other simple types to native python types
                            try:
                                if hasattr(val, 'tolist'):
                                    val = val.tolist()
                                elif isinstance(val, (np.integer, np.floating)):
                                    val = val.item()
                            except Exception:
                                pass
                            settings_snapshot[name] = val
                except Exception:
                    settings_snapshot = None

                resume_info = {
                    'timestamp': datetime.now().isoformat(),
                    'resume_dir': args.resume_dir,
                    'resume_tag': args.resume_tag,
                    'model_path': model_path if os.path.exists(model_path) else None,
                    'obs_rms_path': obs_path if os.path.exists(obs_path) else None,
                    'cmd_args': vars(args),
                    'settings': settings_snapshot,
                    'curriculum': {
                        'ranges': [list(rng) for rng in PERTURB_CURRICULUM_RANGES],
                        'stage_episodes': int(CURRICULUM_STAGE_EPISODES),
                    },
                }
                info_path = os.path.join(save_root, 'resume_info.json')
                with open(info_path, 'w', encoding='utf-8') as fh:
                    json.dump(resume_info, fh, indent=2)
                print(f"Saved resume info and copies to {resume_out} and {info_path}")
            except Exception as copy_err:
                print(f"Warning: failed to record resume provenance: {copy_err}")
        except Exception as resume_err:
            print(f"Warning: failed to resume from {args.resume_dir}: {resume_err}")

    def save_model_and_stats(model_path_suffix, best=False):
        model_filename = f"box_ppo_model_{model_path_suffix}.pt"
        rms_filename = f"obs_rms_{model_path_suffix}.npz"

        model_path = os.path.join(save_root, model_filename)
        rms_path = os.path.join(save_root, rms_filename)

        torch.save(agent.network.state_dict(), model_path)
        if obs_rms is not None:
            mean = obs_rms.mean
            var = obs_rms.var
            count = obs_rms.count
        else:
            mean = np.zeros(OBS_DIM, dtype=np.float64)
            var = np.ones(OBS_DIM, dtype=np.float64)
            count = 1.0
        np.savez(rms_path, mean=mean, var=var, count=count)

        os.makedirs(latest_dir, exist_ok=True)
        shutil.copy(model_path, os.path.join(latest_dir, model_filename))
        shutil.copy(rms_path, os.path.join(latest_dir, rms_filename))
        return model_path, rms_path

    def run_validation_pass(label, model_path, rms_path, total_steps, episodes=1):
        try:
            from validate_best import run_validation

            validation_base_dir = os.path.join(save_root, "validation")
            os.makedirs(validation_base_dir, exist_ok=True)

            file_prefix = f"{label}_step_{total_steps}"
            val_rewards, fig_paths = run_validation(
                model_path,
                rms_path,
                episodes=episodes,
                render=False,
                max_steps=MAX_EPISODE_STEPS,
                save_dir=validation_base_dir,
                file_prefix=file_prefix,
            )

            if val_rewards:
                val_reward = float(val_rewards[0])
                final_plot_path = None
                if fig_paths:
                    src_path = fig_paths[0]
                    dst_name = f"{file_prefix}_reward_{val_reward:.2f}.png"
                    dst_path = os.path.join(validation_base_dir, dst_name)
                    try:
                        if src_path != dst_path:
                            os.replace(src_path, dst_path)
                        final_plot_path = dst_path
                    except Exception:
                        final_plot_path = src_path
                if final_plot_path:
                    print(f"[Validation-{label}] Primary validation plot: {final_plot_path}")
                else:
                    print(f"[Validation-{label}] Reward: {val_reward:.2f}")
            else:
                print(f"[Validation-{label}] Automatic validation produced no reward data.")
        except Exception as val_err:
            print(f"Warning: automatic validation for {label} failed: {val_err}")

    obs = env.reset()
    obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
    if obs_rms is not None:
        obs_rms.update(obs)
        obs_norm = obs_rms.normalize(obs)
        obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
    else:
        obs_norm = obs

    state_stacker.reset(initial_obs=obs_norm)
    state = state_stacker.get_state()

    curriculum_ranges = list(PERTURB_CURRICULUM_RANGES) if PERTURB_CURRICULUM_RANGES else []
    if not curriculum_ranges:
        curriculum_ranges = [(0.0, float(PERTURB_FORCE_SCALE))]
    curriculum_stage = 0
    curriculum_stage_len = max(1, int(CURRICULUM_STAGE_EPISODES))
    curriculum_transitions = []

    def describe_curriculum_range(range_pair):
        min_raw, max_raw = range_pair
        min_abs = abs(float(min_raw))
        max_abs = abs(float(max_raw))
        if max_abs < min_abs:
            min_abs, max_abs = max_abs, min_abs
        if max_abs == 0.0:
            return "[0.0]"
        if min_abs <= 0.0:
            return f"[{-max_abs:.1f}, {max_abs:.1f}]"
        return (
            f"[{-max_abs:.1f}, {-min_abs:.1f}] U "
            f"[{min_abs:.1f}, {max_abs:.1f}]"
        )

    def get_curriculum_range(stage_idx=None):
        idx = curriculum_stage if stage_idx is None else stage_idx
        idx = max(0, min(idx, len(curriculum_ranges) - 1))
        return curriculum_ranges[idx]

    def maybe_advance_curriculum(total_completed_episodes):
        nonlocal curriculum_stage
        completed_stages = []
        while (
            curriculum_stage < len(curriculum_ranges) - 1
            and total_completed_episodes >= (curriculum_stage + 1) * curriculum_stage_len
        ):
            completed_stage_number = curriculum_stage + 1
            curriculum_stage += 1
            new_range = get_curriculum_range()
            stage_number = curriculum_stage + 1
            curriculum_transitions.append((total_completed_episodes, stage_number))
            for env_idx in range(NUM_ENVS):
                perturb_forces[env_idx] = _sample_startup_perturbation(new_range)
            print(
                f"[Curriculum] Advanced to stage {curriculum_stage + 1}/{len(curriculum_ranges)} "
                f"with perturb range {describe_curriculum_range(new_range)} "
                f"at episode {total_completed_episodes}"
            )
            completed_stages.append((completed_stage_number, total_completed_episodes))
        return completed_stages

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_steps = np.zeros(NUM_ENVS, dtype=np.int32)
    episode_counts = np.zeros(NUM_ENVS, dtype=np.int64)
    perturb_forces = np.vstack([
        _sample_startup_perturbation(get_curriculum_range()) for _ in range(NUM_ENVS)
    ])
    perturb_steps_remaining = np.zeros(NUM_ENVS, dtype=np.int32)
    perturb_started = np.zeros(NUM_ENVS, dtype=bool)

    completed_episode_rewards = []
    best_episode_reward = -1e9
    total_steps = 0
    global_episode_counter = 0

    print(
        f"[Curriculum] Starting at stage {curriculum_stage + 1}/{len(curriculum_ranges)} "
        f"with perturb range {describe_curriculum_range(get_curriculum_range())}"
    )

    print(f"Starting PPO training with MJX across {NUM_ENVS} environments...")

    while total_steps < MAX_TRAINING_STEPS:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            sampled_actions, _, _, _ = agent.network.get_action_and_value(state_tensor)
        actions = np.asarray(sampled_actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, np.newaxis]
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)

        allow_actions = episode_steps >= ACTION_START_DELAY
        if np.any(~allow_actions):
            actions = actions.copy()
            actions[~allow_actions, :] = 0.0

        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, log_prob_tensor, _, value_tensor = agent.network.get_action_and_value(state_tensor, action_tensor)

        log_prob_batch = log_prob_tensor.detach().cpu().numpy()
        value_batch = value_tensor.detach().cpu().numpy()
        if value_batch.ndim == 2 and value_batch.shape[1] == 1:
            value_batch = value_batch[:, 0]

        action_scaled = np.clip(actions, -1.0, 1.0) * ACTION_FORCE_MAGNITUDE
        action_forces = action_scaled[:, 0]

        start_mask = (~perturb_started) & (episode_steps >= PERTURB_START_DELAY)
        if np.any(start_mask):
            perturb_steps_remaining[start_mask] = PERTURB_APPLY_STEPS
            perturb_started[start_mask] = True

        perturb_batch = np.zeros((NUM_ENVS, 6), dtype=np.float32)
        active_mask = perturb_steps_remaining > 0
        if np.any(active_mask):
            perturb_batch[active_mask] = perturb_forces[active_mask]
            perturb_steps_remaining[active_mask] -= 1

        obs_features, contact_batch, tilt_angles, tilt_rates, xpos = env.step(action_forces, perturb_batch)
        out_of_bounds = (np.abs(xpos[:, 0]) > BOUNDARY_LIMIT) | (np.abs(xpos[:, 1]) > BOUNDARY_LIMIT)
        rewards, alive_mask = _compute_reward_batch(contact_batch, tilt_angles, tilt_rates, action_forces, out_of_bounds, TILT_LIMIT_RADIANS)

        next_episode_steps = episode_steps + 1
        timeout_mask = next_episode_steps >= MAX_EPISODE_STEPS
        done_mask = (~alive_mask) | timeout_mask

        episode_rewards += rewards
        total_steps += NUM_ENVS

        obs_batch = np.nan_to_num(obs_features, nan=0.0, posinf=1e6, neginf=-1e6)
        if obs_rms is not None:
            obs_rms.update(obs_batch)
            obs_norm = obs_rms.normalize(obs_batch)
            invalid_mask = ~np.isfinite(obs_norm).all(axis=1)
            if np.any(invalid_mask):
                print(f"[WARN] Non-finite normalized observation in envs {np.where(invalid_mask)[0]}; forcing reset.")
                done_mask = np.logical_or(done_mask, invalid_mask)
            obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            obs_norm = obs_batch

        state_stacker.push(obs_norm)
        next_state = state_stacker.get_state()

        stored = agent.store_transition(state, actions, rewards, done_mask.astype(np.float32), log_prob_batch, value_batch)
        if stored:
            agent.update(next_state, done_mask.astype(np.float32))

        state = next_state

        episode_steps = np.where(done_mask, 0, next_episode_steps)
        perturb_started[done_mask] = False
        perturb_steps_remaining[done_mask] = 0

        done_indices = np.where(done_mask)[0]
        if done_indices.size > 0:
            done_rewards = episode_rewards[done_indices].copy()
            episode_counts[done_indices] += 1
            for idx, ep_reward in zip(done_indices, done_rewards):
                completed_episode_rewards.append(float(ep_reward))
                global_episode_counter += 1
                stage_completion_events = maybe_advance_curriculum(global_episode_counter)
                if stage_completion_events:
                    for completed_stage_number, completion_episode in stage_completion_events:
                        stage_suffix = f"stage{completed_stage_number:02d}"
                        stage_model_path, stage_rms_path = save_model_and_stats(stage_suffix)
                        print(
                            f"[Curriculum] Stage {completed_stage_number} completed at episode {completion_episode}; "
                            f"saved snapshot to {stage_model_path}"
                        )
                        run_validation_pass(stage_suffix, stage_model_path, stage_rms_path, total_steps)
                    best_episode_reward = -1e9
                    print(
                        f"[Curriculum] Reset best_episode_reward for new stage "
                        f"({curriculum_stage + 1}/{len(curriculum_ranges)})"
                    )
                if ep_reward > best_episode_reward:
                    best_episode_reward = ep_reward
                    best_model_path, best_rms_path = save_model_and_stats("best", best=True)
                    print(f"New best model saved ({save_root}) (episode {global_episode_counter}) with reward {ep_reward:.2f}")
                    run_validation_pass("best", best_model_path, best_rms_path, total_steps)

            resample_indices = [idx for idx in done_indices if episode_counts[idx] % PERTURB_RESAMPLE_EPISODES == 0]
            for idx in resample_indices:
                perturb_forces[idx] = _sample_startup_perturbation(get_curriculum_range())

            obs_after_reset = env.reset(done_indices)
            reset_obs = np.nan_to_num(obs_after_reset[done_indices], nan=0.0, posinf=1e6, neginf=-1e6)
            if obs_rms is not None:
                obs_rms.update(reset_obs)
                reset_norm = obs_rms.normalize(reset_obs)
                reset_norm = np.nan_to_num(reset_norm, nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                reset_norm = reset_obs

            state_stacker.reset(initial_obs=reset_norm, env_indices=done_indices)
            episode_rewards[done_indices] = 0.0
            perturb_started[done_indices] = False
            perturb_steps_remaining[done_indices] = 0
            state = state_stacker.get_state()

            if global_episode_counter % 20 == 0:
                recent = completed_episode_rewards[-20:]
                recent_mean = float(np.mean(recent)) if recent else 0.0
                print(f"Episode {global_episode_counter} | Total Steps: {total_steps} | Reward: {float(done_rewards[-1]):.2f} | Recent mean: {recent_mean:.2f}")

        try:
            PERIODIC_SAVE_STEPS = int(os.environ.get("PERIODIC_SAVE_STEPS", "1000000"))
            if PERIODIC_SAVE_STEPS > 0 and (total_steps % PERIODIC_SAVE_STEPS == 0):
                last_model_name = f"box_ppo_model_last_{total_steps}.pt"
                last_model_path = os.path.join(save_root, last_model_name)
                torch.save(agent.network.state_dict(), last_model_path)
                shutil.copy(last_model_path, os.path.join(save_root, "box_ppo_model_last.pt"))
                os.makedirs(latest_dir, exist_ok=True)
                shutil.copy(last_model_path, os.path.join(latest_dir, last_model_name))
                shutil.copy(os.path.join(save_root, "box_ppo_model_last.pt"), os.path.join(latest_dir, "box_ppo_model_last.pt"))
        except Exception as e:
            print(f"Warning: periodic checkpoint failed at step {total_steps}: {e}")

    print("Training finished!")

    try:
        torch.save(agent.network.state_dict(), os.path.join(save_root, "box_ppo_model.pt"))
        save_model_and_stats("last")

        best_file = os.path.join(save_root, "box_ppo_model_best.pt")
        if not os.path.exists(best_file):
            shutil.copy(os.path.join(save_root, "box_ppo_model_last.pt"), best_file)
            shutil.copy(os.path.join(save_root, "obs_rms_last.npz"), os.path.join(save_root, "obs_rms_best.npz"))
            shutil.copy(best_file, os.path.join(latest_dir, "box_ppo_model_best.pt"))
            shutil.copy(os.path.join(save_root, "obs_rms_best.npz"), os.path.join(latest_dir, "obs_rms_best.npz"))
            print(f"No best model found; saved final model as {best_file} and obs_rms_best.npz for validation.")
    except Exception as e:
        print(f"Warning: could not save fallback best model or obs_rms: {e}")

    if completed_episode_rewards:
        plt.figure(figsize=(10, 5))
        episodes = np.arange(1, len(completed_episode_rewards) + 1)
        plt.plot(episodes, completed_episode_rewards, label="Episode Reward")
        plt.title("Episode Rewards over Time")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        if curriculum_transitions:
            ymin, ymax = plt.ylim()
            if ymax <= ymin:
                ymax = ymin + 1.0
            text_y = ymax - 0.02 * (ymax - ymin)
            marker_label_used = False
            for raw_episode_idx, stage_number in curriculum_transitions:
                episode_idx = min(max(int(raw_episode_idx), 1), len(completed_episode_rewards))
                line_label = "Curriculum stage change" if not marker_label_used else None
                plt.axvline(
                    x=episode_idx,
                    color="orange",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                    label=line_label,
                )
                plt.text(
                    episode_idx,
                    text_y,
                    f"Stage {stage_number}",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=8,
                    color="orange",
                )
                marker_label_used = True
        plt.legend()
        plt.savefig(os.path.join(save_root, "box_ppo_rewards.png"))
        print(f"Saved reward plot to {os.path.join(save_root, 'box_ppo_rewards.png')}")
    else:
        print("No episode rewards recorded; skipping reward plot.")

    if os.environ.get("SKIP_EVAL", "0") == "1":
        print("Skipping evaluation viewer because SKIP_EVAL=1.")
        return

    print("Starting evaluation...")
    import importlib
    try:
        importlib.import_module("mujoco.viewer")
    except Exception as e:
        print(f"Warning: could not import mujoco.viewer: {e}")
    viewer = mujoco.viewer.launch_passive(eval_model, data)
    state_stacker = StateStacker(OBS_DIM, STACK_SIZE)
    state = reset_sim(eval_model, data, state_stacker, obs_rms)
    eval_episode_step = 0
    eval_active_perturb_steps_remaining = 0
    eval_perturb_started = False
    eval_active_perturb_force = _sample_startup_perturbation(
        PERTURB_CURRICULUM_RANGES[-1] if PERTURB_CURRICULUM_RANGES else None
    )
    try:
        action_gid = mujoco.mj_name2id(eval_model, mujoco.mjtObj.mjOBJ_GEOM, 'action_indicator')
        have_action_indicator = True
    except Exception:
        action_gid = None
        have_action_indicator = False

    while viewer.is_running():
        allow_actions = eval_episode_step >= ACTION_START_DELAY
        if allow_actions:
            action_out, _, _ = agent.select_action(state)
            action_arr = np.atleast_1d(np.asarray(action_out).ravel()).astype(np.float32)
        else:
            action_arr = np.zeros(ACTION_DIM, dtype=np.float32)

        action_scaled = np.clip(action_arr, -1.0, 1.0) * ACTION_FORCE_MAGNITUDE
        data.xfrc_applied[:] = 0
        Fx = np_scalar(action_scaled[0])
        data.xfrc_applied[box_body_id][0] = Fx
        data.xfrc_applied[box_body_id][1] = 0.0
        data.xfrc_applied[box_body_id][2] = 0.0
        data.xfrc_applied[box_body_id][3] = 0.0
        data.xfrc_applied[box_body_id][4] = 0.0
        data.xfrc_applied[box_body_id][5] = 0.0

        if (not eval_perturb_started) and (eval_episode_step >= PERTURB_START_DELAY):
            eval_active_perturb_steps_remaining = PERTURB_APPLY_STEPS
            eval_perturb_started = True

        if eval_active_perturb_steps_remaining > 0:
            data.xfrc_applied[box_body_id] = data.xfrc_applied[box_body_id] + eval_active_perturb_force
            eval_active_perturb_steps_remaining -= 1

        if have_action_indicator:
            try:
                ax = np_scalar(np.clip(action_scaled[0] / (ACTION_FORCE_MAGNITUDE + 1e-8), -1.0, 1.0))
                r = (ax + 1.0) * 0.5
                g = 0.0
                b = 0.0
                a = 0.9
                eval_model.geom_rgba[4 * action_gid:4 * action_gid + 4] = np.array([r, g, b, a], dtype=np.float32)
            except Exception:
                pass

        mujoco.mj_step(eval_model, data)

        obs_eval = get_observation(data)
        obs_eval = np.nan_to_num(obs_eval, nan=0.0, posinf=1e6, neginf=-1e6)
        if obs_rms is not None:
            obs_norm_eval = obs_rms.normalize(obs_eval)
        else:
            obs_norm_eval = obs_eval
        obs_norm_eval = np.nan_to_num(obs_norm_eval, nan=0.0, posinf=1e6, neginf=-1e6)
        state_stacker.push(obs_norm_eval)
        state = state_stacker.get_state()

        pos = np.asarray(data.body('box_body').xpos, dtype=np.float32)
        eval_out_of_bounds = (abs(pos[0]) > BOUNDARY_LIMIT) or (abs(pos[1]) > BOUNDARY_LIMIT)
        if eval_out_of_bounds:
            reset_sim(eval_model, data, state_stacker, obs_rms)
            eval_episode_step = 0
            eval_active_perturb_steps_remaining = 0
            eval_perturb_started = False
            eval_active_perturb_force = _sample_startup_perturbation(
                PERTURB_CURRICULUM_RANGES[-1] if PERTURB_CURRICULUM_RANGES else None
            )
            continue
        if is_done(data):
            reset_sim(eval_model, data, state_stacker, obs_rms)
            eval_episode_step = 0
            eval_active_perturb_steps_remaining = 0
            eval_perturb_started = False
            eval_active_perturb_force = _sample_startup_perturbation(
                PERTURB_CURRICULUM_RANGES[-1] if PERTURB_CURRICULUM_RANGES else None
            )
        else:
            eval_episode_step += 1

        viewer.sync()
        time.sleep(0.02)

    viewer.close()

if __name__ == "__main__":
    main()
