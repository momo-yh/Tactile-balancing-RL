import os
import argparse
from datetime import datetime

import mujoco
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse core classes/utilities from the current training script
from train_box_ppo import (
    ActorCritic,
    StateStacker,
    RunningMeanStd,
    get_observation,
    np_scalar,
)
from reward import compute_reward
from settings import (
    STACK_SIZE, OBS_DIM, STATE_DIM, ACTION_DIM, HIDDEN_DIM,
    ACTION_FORCE_MAGNITUDE, PERTURB_START_DELAY, PERTURB_RESAMPLE_EPISODES,
    PERTURB_APPLY_STEPS, PERTURB_FORCE_SCALE,
    BOUNDARY_LIMIT, TILT_LIMIT_RADIANS, USE_OBS_NORMALIZATION,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ou_step(x, theta, mu, sigma):
    # Discrete-time OU update with dt=1
    # x_{t+1} = x_t + theta*(mu - x_t) + sigma*N(0,1)
    return x + theta * (mu - x) + sigma * np.random.randn()


def run_validation_random(
    model_path: str,
    obs_rms_path: str | None = None,
    episodes: int = 5,
    render: bool = False,
    max_steps: int = 1000,
    save_dir: str | None = None,
    file_prefix: str | None = None,
    noise_type: str = "gaussian",  # gaussian | uniform | ou
    noise_scale: float = 5.0,       # scale for gaussian sigma or uniform range/2
    ou_theta: float = 0.05,
    ou_mu: float = 0.0,
    seed: int | None = None,
    perturb_interval: int | None = None,
    perturb_duration: int = 1,
    compare_open_loop: bool = False,
):
    if seed is not None:
        base_seed = seed
    else:
        base_seed = None

    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    box_body_id = data.body('box_body').id

    net = ActorCritic(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # Load obs normalization if enabled
    if USE_OBS_NORMALIZATION:
        obs_rms = RunningMeanStd(shape=(OBS_DIM,))
        if obs_rms_path is not None and os.path.exists(obs_rms_path):
            d = np.load(obs_rms_path)
            obs_rms.mean = d['mean']
            obs_rms.var = d['var']
            obs_rms.count = d['count']
        else:
            print("No obs_rms provided; running validation without loaded stats")
    else:
        obs_rms = None

    state_stacker = StateStacker(OBS_DIM, STACK_SIZE)

    rewards = []
    saved_figs = []

    viewer = None
    if render:
        try:
            import importlib
            importlib.import_module("mujoco.viewer")
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            print(f"Warning: could not import or launch viewer: {e}")
            render = False

    if save_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join("validation", ts)
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(episodes):
        # reset and get initial observation/state
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        initial_obs = get_observation(data)
        initial_obs = np.nan_to_num(initial_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        if obs_rms is not None:
            normalized = obs_rms.normalize(initial_obs)
            if not np.isfinite(normalized).all():
                print(f"Warning: non-finite normalized reset observation in episode {ep+1}; skipping.")
                continue
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1e6, neginf=-1e6)
            state_stacker.reset(normalized)
        else:
            state_stacker.reset(initial_obs)

        state = state_stacker.get_state()
        if not np.isfinite(state).all():
            print(f"Warning: non-finite stacked reset state in episode {ep+1}; skipping.")
            continue

        # RNG for this episode
        rng = np.random.RandomState(base_seed + ep if base_seed is not None else None)

        # Prepare noise sequence according to spacing
        noise_seq = np.zeros(max_steps, dtype=np.float32)
        if perturb_interval is None or perturb_interval <= 0:
            # continuous per-step noise
            if noise_type == 'gaussian':
                noise_seq = rng.randn(max_steps).astype(np.float32) * float(noise_scale)
            elif noise_type == 'uniform':
                noise_seq = rng.uniform(-float(noise_scale), float(noise_scale), size=max_steps).astype(np.float32)
            elif noise_type == 'ou':
                x = 0.0
                for t in range(max_steps):
                    x = x + float(ou_theta) * (float(ou_mu) - x) + float(noise_scale) * rng.randn()
                    noise_seq[t] = x
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
        else:
            # spaced windows
            for start in range(0, max_steps, int(perturb_interval)):
                end = min(start + int(perturb_duration), max_steps)
                if noise_type == 'gaussian':
                    val = rng.randn() * float(noise_scale)
                    noise_seq[start:end] = val
                elif noise_type == 'uniform':
                    val = rng.uniform(-float(noise_scale), float(noise_scale))
                    noise_seq[start:end] = val
                elif noise_type == 'ou':
                    x = 0.0
                    for t in range(start, end):
                        x = x + float(ou_theta) * (float(ou_mu) - x) + float(noise_scale) * rng.randn()
                        noise_seq[t] = x
                else:
                    raise ValueError(f"Unknown noise_type: {noise_type}")

        # optional startup perturb force (same semantics as training)
        startup_force = np.zeros(6, dtype=np.float32)
        if ep % PERTURB_RESAMPLE_EPISODES == 0:
            Fx_pf = rng.uniform(-PERTURB_FORCE_SCALE, PERTURB_FORCE_SCALE)
            startup_force[0] = Fx_pf

        # capture initial sim arrays to restore exact starting state for comparison
        init_sim = {
            'qpos': data.qpos.copy(),
            'qvel': data.qvel.copy(),
            'act': data.act.copy() if hasattr(data, 'act') else None,
        }

        modes = ['controller']
        if compare_open_loop:
            modes.append('openloop')

        ep_results = {}
        for mode in modes:
            # restore sim and stacker
            data.qpos[:] = init_sim['qpos']
            data.qvel[:] = init_sim['qvel']
            if init_sim['act'] is not None:
                data.act[:] = init_sim['act']
            mujoco.mj_forward(model, data)
            state_stacker.reset(initial_obs if obs_rms is None else obs_rms.normalize(initial_obs))
            state = state_stacker.get_state()

            ep_reward = 0.0
            steps = 0
            episode_step = 0
            active_startup_remaining = 0
            perturb_started = False

            actions_per_step = []
            noise_per_step = []
            combined_per_step = []
            sensors_per_step = []
            rewards_per_step = []

            while steps < max_steps:
                allow_actions = episode_step >= PERTURB_START_DELAY
                if mode == 'controller' and allow_actions:
                    with torch.no_grad():
                        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                        if state_t.dim() == 1:
                            state_t = state_t.unsqueeze(0)
                        mean = net.actor(state_t)
                        action_t = torch.tanh(mean)
                        action = np.atleast_1d(action_t.cpu().numpy().ravel().astype(np.float32))
                else:
                    action = np.zeros(ACTION_DIM, dtype=np.float32)

                action_scaled = np.clip(action, -1.0, 1.0) * ACTION_FORCE_MAGNITUDE

                noise_fx = float(noise_seq[steps])

                data.xfrc_applied[:] = 0
                Fx = np_scalar(action_scaled[0]) + noise_fx
                data.xfrc_applied[box_body_id][0] = Fx

                if (not perturb_started) and (episode_step >= PERTURB_START_DELAY):
                    active_startup_remaining = PERTURB_APPLY_STEPS
                    perturb_started = True
                if active_startup_remaining > 0:
                    data.xfrc_applied[box_body_id] = data.xfrc_applied[box_body_id] + startup_force
                    active_startup_remaining -= 1

                mujoco.mj_step(model, data)
                steps += 1
                episode_step += 1

                obs = get_observation(data)
                obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
                sensors_per_step.append([float(obs[0]), float(obs[1]), float(obs[2])])

                pos = np.asarray(data.body('box_body').xpos, dtype=np.float32)
                out_of_bounds = (abs(pos[0]) > BOUNDARY_LIMIT) or (abs(pos[1]) > BOUNDARY_LIMIT)

                if obs_rms is not None:
                    obs_norm = obs_rms.normalize(obs)
                    if not np.isfinite(obs_norm).all():
                        print(f"Warning: non-finite normalized observation in episode {ep+1}; aborting episode.")
                        break
                    obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
                else:
                    obs_norm = obs
                state_stacker.push(obs_norm)
                state = state_stacker.get_state()
                if not np.isfinite(state).all():
                    print(f"Warning: non-finite stacked state in episode {ep+1}; aborting episode.")
                    break

                reward, alive = compute_reward(data, Fx, TILT_LIMIT_RADIANS, out_of_bounds=out_of_bounds)
                if not alive:
                    done = True
                else:
                    done = False

                ep_reward += reward

                actions_per_step.append(float(action_scaled[0]))
                noise_per_step.append(float(noise_fx))
                combined_per_step.append(float(Fx))
                rewards_per_step.append(float(reward))

                if render and viewer is not None and viewer.is_running():
                    viewer.sync()

                if done:
                    break

            ep_results[mode] = {
                'reward': ep_reward,
                'actions': np.array(actions_per_step, dtype=np.float32),
                'noise': np.array(noise_per_step, dtype=np.float32),
                'combined': np.array(combined_per_step, dtype=np.float32),
                'sensors': np.array(sensors_per_step, dtype=np.float32),
                'rewards': np.array(rewards_per_step, dtype=np.float32),
            }

        # save outputs per mode
        for mode, data_dict in ep_results.items():
            ep_reward = data_dict['reward']
            rewards.append(ep_reward)
            print(f"Random-perturb validation ep {ep+1} mode={mode} reward: {ep_reward:.2f} steps: {len(data_dict['combined'])}")

            try:
                xs = np.arange(1, len(data_dict['combined']) + 1)
                actions_arr = data_dict['actions']
                noise_arr = data_dict['noise']
                sensors_arr = data_dict['sensors']
                rewards_arr = data_dict['rewards']

                # Plot actions, noise, sensors, and reward in separate subplots (keep sensor data)
                fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

                # Actions (agent-applied force)
                axs[0].plot(xs, actions_arr, label='agent action (force)', color='tab:blue')
                axs[0].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
                axs[0].legend()
                axs[0].set_ylabel('Action (Fx)')

                # Noise (external perturbation)
                axs[1].plot(xs, noise_arr, label=f'noise ({noise_type})', color='tab:orange')
                axs[1].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
                axs[1].legend()
                axs[1].set_ylabel('Noise (Fx)')

                # Sensor readings (if available)
                if sensors_arr.size > 0:
                    # sensors_arr expected shape (T, 3)
                    axs[2].plot(xs, sensors_arr[:, 0], label='sensor_p1_0', color='tab:green')
                    axs[2].plot(xs, sensors_arr[:, 1], label='sensor_0_0', color='tab:red')
                    axs[2].plot(xs, sensors_arr[:, 2], label='sensor_m1_0', color='tab:cyan')
                    axs[2].legend()
                else:
                    axs[2].text(0.5, 0.5, 'No sensor data', ha='center', va='center', transform=axs[2].transAxes)
                axs[2].set_ylabel('Sensors')

                # Reward per step
                axs[3].plot(xs, rewards_arr, color='tab:purple', label='reward per step')
                axs[3].legend()
                axs[3].set_ylabel('Reward')
                axs[3].set_xlabel('Step')

                fig.suptitle(f'Random Perturbation Validation (ep {ep+1}) mode={mode}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                name = (file_prefix or f"random_perturb") + f"_{mode}"
                out_path = os.path.join(save_dir, f"{name}_ep_{ep+1}.png")
                fig.savefig(out_path)
                plt.close(fig)
                saved_figs.append(out_path)
                print(f"Saved random-perturb plot to {out_path}")
            except Exception as e:
                print(f"Warning: failed to save random-perturb plot: {e}")

    if viewer is not None:
        viewer.close()

    avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else float('nan')
    print(f"Random-perturb validation completed. Average reward over {episodes} episodes: {avg_reward:.2f}")
    return rewards, saved_figs


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='pretrained/latest/box_ppo_model_best.pt', help='Path to model weights')
    p.add_argument('--obs_rms', type=str, default='pretrained/latest/obs_rms_best.npz', help='Path to obs_rms .npz file')
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--render', action='store_true')
    p.add_argument('--max_steps', type=int, default=1000)
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--file_prefix', type=str, default=None)
    p.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian', 'uniform', 'ou'])
    p.add_argument('--noise_scale', type=float, default=5.0, help='Std for gaussian, half-range for uniform, sigma for OU')
    p.add_argument('--ou_theta', type=float, default=0.05)
    p.add_argument('--ou_mu', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--perturb_interval', type=int, default=None,
                   help='Spacing (in steps) between perturbation windows. None or 0 => continuous per-step noise')
    p.add_argument('--perturb_duration', type=int, default=1,
                   help='Duration (in steps) of each perturbation window when perturb_interval is set')
    p.add_argument('--compare_open_loop', action='store_true', help='Also run an open-loop (zero-action) replay for comparison')
    args = p.parse_args()

    run_validation_random(
        model_path=args.model,
        obs_rms_path=args.obs_rms,
        episodes=args.episodes,
        render=args.render,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        file_prefix=args.file_prefix,
        noise_type=args.noise_type,
        noise_scale=args.noise_scale,
        ou_theta=args.ou_theta,
        ou_mu=args.ou_mu,
        seed=args.seed,
        perturb_interval=args.perturb_interval,
        perturb_duration=args.perturb_duration,
        compare_open_loop=args.compare_open_loop,
    )
