import argparse
import mujoco
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
from reward import compute_reward
from settings import (
    STACK_SIZE, OBS_DIM, STATE_DIM, ACTION_DIM, HIDDEN_DIM,
    ACTION_FORCE_MAGNITUDE, PERTURB_FORCE_SCALE, PERTURB_APPLY_STEPS,
    PERTURB_START_DELAY, PERTURB_RESAMPLE_EPISODES, ACTION_START_DELAY,
    BOUNDARY_LIMIT, TILT_LIMIT_RADIANS, USE_OBS_NORMALIZATION
)
from features import build_observation, compute_tilt_angle

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
            nn.Linear(hidden_dim, action_dim)
        )

        # learnable log std (initialized near 0)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
    """Storage for PPO rollout data (supports continuous actions)."""
    def __init__(self, rollout_length, obs_dim, action_dim, gamma, gae_lambda):
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((rollout_length, obs_dim), dtype=np.float32)
        # store continuous actions (one entry per action dim)
        self.actions = np.zeros((rollout_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)

        self.step = 0

    def store(self, obs, action, reward, done, log_prob, value):
        # Guard against accidental overflow: if step reached rollout_length,
        # reset to 0 and warn. This avoids IndexError when surrounding logic
        # may occasionally attempt one extra store. Ideally the caller
        # should only store exactly `rollout_length` steps between updates.
        if self.step >= self.rollout_length:
            print(f"[RolloutBuffer] Warning: buffer index {self.step} >= rollout_length {self.rollout_length}. Resetting step to 0 and overwriting.")
            self.step = 0

        self.observations[self.step] = obs
        # action may be scalar or array-like; coerce to float array
        try:
            a = np.asarray(action, dtype=np.float32).ravel()
            # if action has multiple dims, take first action_dim entries
            self.actions[self.step, :a.size] = a
        except Exception:
            # fallback: try to cast to float
            try:
                self.actions[self.step, 0] = float(action)
            except Exception:
                self.actions[self.step, :] = 0.0
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        # log_prob may be a torch tensor or numpy scalar
        try:
            self.log_probs[self.step] = log_prob.item()
        except Exception:
            self.log_probs[self.step] = float(log_prob)
        # always write value (tensor or scalar) and increment step
        try:
            self.values[self.step] = value.item()
        except Exception:
            try:
                self.values[self.step] = float(value)
            except Exception:
                self.values[self.step] = 0.0
        self.step += 1

    def compute_advantages(self, last_value, last_done):
        """Compute advantages using GAE."""
        gae = 0
        last_value = last_value.item()
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
        """Yield minibatches for PPO updates."""
        indices = np.random.permutation(self.rollout_length)
        for start in range(0, self.rollout_length, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]
            obs_b = torch.tensor(self.observations[mb_indices], dtype=torch.float32, device=device)
            actions_b = torch.tensor(self.actions[mb_indices], dtype=torch.float32, device=device)

            yield (
                obs_b,
                actions_b,
                torch.tensor(self.log_probs[mb_indices], dtype=torch.float32, device=device),
                torch.tensor(self.advantages[mb_indices], dtype=torch.float32, device=device),
                torch.tensor(self.returns[mb_indices], dtype=torch.float32, device=device)
            )

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4, 
                 rollout_length=2048, batch_size=64, ppo_epochs=10, 
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        # action_dim here is the network output dim; if discrete=True this should be the number of discrete actions
        self.network = ActorCritic(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = None

        self.buffer = RolloutBuffer(rollout_length, state_dim, action_dim, gamma, gae_lambda)

        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.step_count = 0
        self.rollout_length = rollout_length
        self.update_count = 0

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def select_action(self, state):
        """Select an action from the policy (returns numpy action for env)."""
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state)
        # If training path returned a tensor action, convert to numpy for environment use
        if torch.is_tensor(action):
            action_np = action.cpu().numpy()
        else:
            action_np = np.asarray(action)
        return action_np, log_prob, value

    def store_transition(self, *args):
        """Store a single transition."""
        obs, action, reward, done, log_prob, value = args

        obs_arr = np.asarray(obs, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32)
        lp = float(log_prob.item() if torch.is_tensor(log_prob) else log_prob)
        val = float(value.item() if torch.is_tensor(value) else value)

        if not np.isfinite(obs_arr).all() or not np.isfinite(action_arr).all() or not np.isfinite(lp) or not np.isfinite(val):
            print("[WARN] Skipping transition with non-finite data (obs/action/log_prob/value).")
            return False

        # Replace with sanitized versions to avoid lingering -inf/inf
        obs = np.nan_to_num(obs_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        action = np.nan_to_num(action_arr, nan=0.0, posinf=1.0, neginf=-1.0)

        self.buffer.store(obs, action, reward, done, lp, val)
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
        self.buffer.compute_advantages(last_value, last_done)
        
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages
        
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

    # 1. Load Model
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    box_body_id = data.body('box_body').id

    # 2. Hyperparameters
    # Use shared configuration from settings.py for these parameters
    # (STACK_SIZE, OBS_DIM, ACTION_DIM, etc.)
    # Local training-only parameters below:
    MAX_TRAINING_STEPS = 5000000
    ROLLOUT_LENGTH = 512
    BATCH_SIZE = 128     # smaller minibatch to increase gradient variance
    PPO_EPOCHS = 16     # more epochs per rollout to better exploit good rollouts
    LR = 1e-4
    ENTROPY_COEF = 0.01
    LR_DECAY_FRACTION = float(os.environ.get("LR_DECAY_FRACTION", "0.75"))
    LR_DECAY_GAMMA = float(os.environ.get("LR_DECAY_GAMMA", "0.1"))
    # Episode length limit (when reached the episode ends)
    MAX_EPISODE_STEPS = 2000
    # Short test mode: set SHORT_TEST=1 in env to run a much shorter training for quick checks
    if os.environ.get("SHORT_TEST", "0") == "1":
        MAX_TRAINING_STEPS = 20000
        ROLLOUT_LENGTH = 256
    # Prepare timestamped save directory under pretrained/<timestamp> and a 'latest' mirror
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tilt_deg = int(round(np.degrees(TILT_LIMIT_RADIANS)))
    save_root = os.path.join("pretrained", f"{ts}_TILT{tilt_deg}_RL")
    latest_dir = os.path.join("pretrained", "latest")
    os.makedirs(save_root, exist_ok=True)
    
    # 3. Initialize Agent and helpers
    # The network action_dim is the continuous action dimension
    agent = PPOAgent(STATE_DIM, ACTION_DIM, hidden_dim=HIDDEN_DIM,
                     rollout_length=ROLLOUT_LENGTH, batch_size=BATCH_SIZE, ppo_epochs=PPO_EPOCHS,
                     lr=LR, gamma=0.99, gae_lambda=0.95, entropy_coef=ENTROPY_COEF)

    if LR_DECAY_GAMMA < 0.999:
        LR_DECAY_FRACTION = min(max(LR_DECAY_FRACTION, 0.0), 1.0)
        expected_updates = max(1, MAX_TRAINING_STEPS // ROLLOUT_LENGTH)
        decay_update = max(1, int(expected_updates * LR_DECAY_FRACTION))
        scheduler = MultiStepLR(agent.optimizer, milestones=[decay_update], gamma=LR_DECAY_GAMMA)
        agent.set_scheduler(scheduler)
        print(f"[LR Scheduler] MultiStepLR enabled: decay after {decay_update} updates (≈{expected_updates} total), gamma={LR_DECAY_GAMMA}")

    state_stacker = StateStacker(OBS_DIM, STACK_SIZE)
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
                print(f"Warning: observation stats file missing at {obs_path}" )

            print(f"Resumed model weights from {model_path}")
        except Exception as resume_err:
            print(f"Warning: failed to resume from {args.resume_dir}: {resume_err}")
    
    # Helper to save model and obs_rms stats (defined once so it's available throughout main)
    def save_model_and_stats(model_path_suffix, best=False):
        """Helper to save model and obs_rms stats. Returns (model_path, rms_path)."""
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

        # Also update the 'latest' directory
        os.makedirs(latest_dir, exist_ok=True)
        shutil.copy(model_path, os.path.join(latest_dir, model_filename))
        shutil.copy(rms_path, os.path.join(latest_dir, rms_filename))
        return model_path, rms_path
    
    # 4. Training Loop
    episode_rewards = []
    # Track best episode reward and save best model + obs_rms
    best_episode_reward = -1e9
    total_steps = 0
    episode_count = 0
    
    state = reset_sim(model, data, state_stacker, obs_rms)
    # per-episode bookkeeping for startup perturbation and time limit
    episode_step = 0
    active_perturb_steps_remaining = 0
    active_perturb_force = np.zeros(6, dtype=np.float32)
    
    print("Starting PPO training...")
    # debug print removed for removed eccentric options
    while total_steps < MAX_TRAINING_STEPS:
        episode_reward_sum = 0.0

        # initialize startup perturbation for the new episode: sample one random x push
        # but delay applying it until episode_step >= PERTURB_START_DELAY
        active_perturb_steps_remaining = 0
        perturb_started = False
        # sample or reuse a startup perturbation force (x only, optionally eccentric)
        # If PERTURB_RESAMPLE_EPISODES > 1 we keep the same perturbation for several episodes
        if episode_count % PERTURB_RESAMPLE_EPISODES == 0:
            pf = np.zeros(6, dtype=np.float32)
            Fx_pf = np.random.uniform(-PERTURB_FORCE_SCALE, PERTURB_FORCE_SCALE)
            pf[0] = Fx_pf
            # ensure no torque components are present (pure-x perturbation)
            pf[3:] = 0.0
            active_perturb_force = pf
        episode_step = 0

        bad_transition = False
        done = False

        while (not done) and (total_steps < MAX_TRAINING_STEPS):
            allow_actions = episode_step >= ACTION_START_DELAY

            if allow_actions:
                action_out, log_prob, value = agent.select_action(state)
                action_arr = np.atleast_1d(np.asarray(action_out).ravel()).astype(np.float32)
            else:
                zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
                _, log_prob, _, value = agent.network.get_action_and_value(state, zero_action)
                action_arr = zero_action

            action_unscaled = action_arr
            action_scaled = np.clip(action_unscaled, -1.0, 1.0) * ACTION_FORCE_MAGNITUDE

            data.xfrc_applied[:] = 0
            Fx = np_scalar(action_scaled[0])
            data.xfrc_applied[box_body_id][0] = Fx
            data.xfrc_applied[box_body_id][1] = 0.0
            data.xfrc_applied[box_body_id][2] = 0.0
            data.xfrc_applied[box_body_id][3] = 0.0
            data.xfrc_applied[box_body_id][4] = 0.0
            data.xfrc_applied[box_body_id][5] = 0.0

            if (not perturb_started) and (episode_step >= PERTURB_START_DELAY):
                active_perturb_steps_remaining = PERTURB_APPLY_STEPS
                perturb_started = True

            if active_perturb_steps_remaining > 0:
                data.xfrc_applied[box_body_id] = data.xfrc_applied[box_body_id] + active_perturb_force
                active_perturb_steps_remaining -= 1

            if os.environ.get("DEBUG_EPISODES", "0") == "1" and episode_count < 5 and (episode_step % 50 == 0):
                try:
                    print(f"[DEBUG] applied xfrc_applied (body={box_body_id}): {data.xfrc_applied[box_body_id]}")
                except Exception:
                    pass

            mujoco.mj_step(model, data)
            total_steps += 1
            episode_step += 1

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

            obs = get_observation(data)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            if obs_rms is not None:
                obs_rms.update(obs)
                obs_norm = obs_rms.normalize(obs)
                if not np.isfinite(obs_norm).all():
                    print(f"[WARN] Non-finite normalized observation at step {total_steps}; resetting episode.")
                    state = reset_sim(model, data, state_stacker, obs_rms)
                    bad_transition = True
                    done = True
                    break
            else:
                obs_norm = obs
            obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
            state_stacker.push(obs_norm)
            next_state = state_stacker.get_state()
            if not np.isfinite(next_state).all():
                print(f"[WARN] Non-finite stacked state at step {total_steps}; resetting episode.")
                state = reset_sim(model, data, state_stacker, obs_rms)
                bad_transition = True
                done = True
                break

            timeout = (episode_step >= MAX_EPISODE_STEPS)
            pos = np.asarray(data.body('box_body').xpos, dtype=np.float32)
            out_of_bounds = (abs(pos[0]) > BOUNDARY_LIMIT) or (abs(pos[1]) > BOUNDARY_LIMIT)

            reward, alive = compute_reward(data, Fx, TILT_LIMIT_RADIANS, out_of_bounds=out_of_bounds)
            done = (not alive) or timeout

            try:
                if os.environ.get("DEBUG_EPISODES", "0") == "1" and episode_count < 5:
                    print(f"[DEBUG] ep={episode_count} step={episode_step} alive={alive} timeout={timeout} reward={reward:.3f}")
            except Exception:
                pass

            episode_reward_sum += reward

            stored = agent.store_transition(state, action_unscaled, reward, float(done), log_prob, value)

            if stored:
                agent.update(state, float(done))

            state = next_state

            if os.environ.get("DEBUG_EPISODE_PROGRESS", "0") == "1":
                try:
                    print(
                        f"[EP {episode_count}] step={episode_step} reward_sum={episode_reward_sum:.2f} "
                        f"done={done} alive={alive} timeout={timeout} out_of_bounds={out_of_bounds}"
                    )
                except Exception:
                    pass

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward_sum)

                try:
                    if episode_reward_sum > best_episode_reward:
                        best_episode_reward = episode_reward_sum
                        best_model_path, best_rms_path = save_model_and_stats("best", best=True)
                        print(f"New best model saved ({save_root}) (episode {episode_count}) with reward {episode_reward_sum:.2f}")

                        try:
                            from validate_best import run_validation

                            validation_base_dir = os.path.join(save_root, "validation")
                            os.makedirs(validation_base_dir, exist_ok=True)

                            file_prefix = f"best_step_{total_steps}"
                            val_rewards, fig_paths = run_validation(
                                best_model_path,
                                best_rms_path,
                                episodes=1,
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
                                    print(f"Primary validation plot: {final_plot_path}")
                            else:
                                print("Automatic validation produced no reward data.")
                        except Exception as val_err:
                            print(f"Warning: automatic validation after best save failed: {val_err}")
                except Exception as e:
                    print(f"Warning: failed to save best model or obs_rms: {e}")

                if episode_count % 20 == 0:
                    print(f"Episode {episode_count} | Total Steps: {total_steps} | Reward: {episode_reward_sum:.2f}")

                state = reset_sim(model, data, state_stacker, obs_rms)
                episode_reward_sum = 0.0
                break

        if bad_transition:
            episode_reward_sum = 0.0
            continue



    # Note: updates are performed inline when the buffer fills. No-op here.

    print("Training finished!")

    try:
        # Save final model and stats
        torch.save(agent.network.state_dict(), os.path.join(save_root, "box_ppo_model.pt"))
        save_model_and_stats("last")

        # If no best model was saved during training, save the final model as the best
        best_file = os.path.join(save_root, "box_ppo_model_best.pt")
        if not os.path.exists(best_file):
            shutil.copy(os.path.join(save_root, "box_ppo_model_last.pt"), best_file)
            shutil.copy(os.path.join(save_root, "obs_rms_last.npz"), os.path.join(save_root, "obs_rms_best.npz"))
            shutil.copy(best_file, os.path.join(latest_dir, "box_ppo_model_best.pt"))
            shutil.copy(os.path.join(save_root, "obs_rms_best.npz"), os.path.join(latest_dir, "obs_rms_best.npz"))
            print(f"No best model found; saved final model as {best_file} and obs_rms_best.npz for validation.")
    except Exception as e:
        print(f"Warning: could not save fallback best model or obs_rms: {e}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Episode Rewards over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(save_root, "box_ppo_rewards.png"))
    print(f"Saved reward plot to {os.path.join(save_root, 'box_ppo_rewards.png')}")

    # --- Evaluation ---
    print("Starting evaluation...")
    # Lazily import the viewer right before use so training (or tests) running headless
    # do not need GUI dependencies loaded at module import time.
    import importlib
    try:
        # import the mujoco.viewer module without binding 'mujoco' in local scope
        importlib.import_module("mujoco.viewer")
    except Exception as e:
        print(f"Warning: could not import mujoco.viewer: {e}")
    viewer = mujoco.viewer.launch_passive(model, data)
    state = reset_sim(model, data, state_stacker, obs_rms)
    # Evaluation: use the same delayed startup perturbation logic per episode
    eval_episode_step = 0
    eval_active_perturb_steps_remaining = 0
    eval_perturb_started = False
    eval_active_perturb_force = np.zeros(6, dtype=np.float32)
    # sample once per episode (will be re-sampled on resets) - x only
    eval_active_perturb_force[0] = np.random.uniform(-PERTURB_FORCE_SCALE, PERTURB_FORCE_SCALE)
    # prepare action indicator geom id (if available)
    try:
        action_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'action_indicator')
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

        action = action_arr
        action_scaled = np.clip(action, -1.0, 1.0) * ACTION_FORCE_MAGNITUDE
        # clear global applied forces first (perturbation uses xfrc_applied)
        data.xfrc_applied[:] = 0
        # Apply agent action exactly like in training: write spatial x-force
        # directly into data.xfrc_applied. (Height-based torque option removed.)
        Fx = np_scalar(action_scaled[0])
        # write pure x force and zero torque components to avoid unintentional moments
        data.xfrc_applied[box_body_id][0] = Fx
        data.xfrc_applied[box_body_id][1] = 0.0
        data.xfrc_applied[box_body_id][2] = 0.0
        data.xfrc_applied[box_body_id][3] = 0.0
        data.xfrc_applied[box_body_id][4] = 0.0
        data.xfrc_applied[box_body_id][5] = 0.0
        # No height-derived torque applied; force is applied at COM only.

        # evaluation delayed-start startup perturbation
        if (not eval_perturb_started) and (eval_episode_step >= PERTURB_START_DELAY):
            eval_active_perturb_steps_remaining = PERTURB_APPLY_STEPS
            eval_perturb_started = True

        if eval_active_perturb_steps_remaining > 0:
            data.xfrc_applied[box_body_id] = data.xfrc_applied[box_body_id] + eval_active_perturb_force
            eval_active_perturb_steps_remaining -= 1
        # update visual action indicator in the viewer (color maps x->red)
        if have_action_indicator:
            try:
                # normalize action to [-1,1] using ACTION_FORCE_MAGNITUDE
                ax = np_scalar(np.clip(action_scaled[0] / (ACTION_FORCE_MAGNITUDE + 1e-8), -1.0, 1.0))
                r = (ax + 1.0) * 0.5
                g = 0.0
                b = 0.0
                a = 0.9
                model.geom_rgba[4*action_gid:4*action_gid+4] = np.array([r, g, b, a], dtype=np.float32)
            except Exception:
                pass

        mujoco.mj_step(model, data)
        
        obs = get_observation(data)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        if obs_rms is not None:
            obs_norm = obs_rms.normalize(obs)
        else:
            obs_norm = obs
        obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
        state_stacker.push(obs_norm)
        state = state_stacker.get_state()

        # evaluation: check out-of-bounds and reset like training
        pos = np.asarray(data.body('box_body').xpos, dtype=np.float32)
        eval_out_of_bounds = (abs(pos[0]) > BOUNDARY_LIMIT) or (abs(pos[1]) > BOUNDARY_LIMIT)
        if eval_out_of_bounds:
            # immediate reset for evaluation
            reset_sim(model, data, state_stacker, obs_rms)
            eval_episode_step = 0
            eval_active_perturb_steps_remaining = 0
            eval_perturb_started = False
            eval_active_perturb_force[0] = np.random.uniform(-PERTURB_FORCE_SCALE, PERTURB_FORCE_SCALE)
            continue
        if is_done(data):
            # reset evaluation episode bookkeeping
            reset_sim(model, data, state_stacker, obs_rms)
            eval_episode_step = 0
            eval_active_perturb_steps_remaining = 0
            eval_perturb_started = False
            eval_active_perturb_force[0] = np.random.uniform(-PERTURB_FORCE_SCALE, PERTURB_FORCE_SCALE)
        else:
            eval_episode_step += 1

        viewer.sync()
        time.sleep(0.02)

    viewer.close()

if __name__ == "__main__":
    main()
