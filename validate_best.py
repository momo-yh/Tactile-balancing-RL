import mujoco
import numpy as np
import os
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime
import argparse, importlib
from train_box_ppo import ActorCritic, StateStacker, RunningMeanStd, get_observation, np_scalar

# Device selection (reuse same logic as training file)
from reward import compute_reward
from settings import (
    STACK_SIZE, OBS_DIM, STATE_DIM, ACTION_DIM, HIDDEN_DIM,
    ACTION_FORCE_MAGNITUDE, PERTURB_FORCE_SCALE, PERTURB_APPLY_STEPS,
    PERTURB_START_DELAY, PERTURB_RESAMPLE_EPISODES, ACTION_START_DELAY, BOUNDARY_LIMIT,
    TILT_LIMIT_RADIANS, USE_OBS_NORMALIZATION,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation script settings (sourced from shared `settings.py`)

def run_validation(
    model_path,
    obs_rms_path=None,
    episodes=10,
    render=False,
    max_steps=1000,
    save_dir=None,
    file_prefix=None,
    force_given=None,
):
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    box_body_id = data.body('box_body').id

    net = ActorCritic(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # Load obs normalization if provided
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

    # If rendering, launch viewer for interactive visualization
    viewer = None
    if render:
        try:
            # import the mujoco.viewer module without binding 'mujoco' in local scope
            importlib.import_module("mujoco.viewer")
        except Exception as e:
            print(f"Warning: could not import mujoco.viewer: {e}")
            render = False # disable rendering if import fails
        viewer = mujoco.viewer.launch_passive(model, data)

    # Determine output directory for plots. When save_dir is provided, reuse it;
    # otherwise create a new timestamped subdirectory under validation/.
    if save_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join("validation", ts)
    os.makedirs(save_dir, exist_ok=True)
    # print(f"Validation plots will be saved to: {save_dir}")

    for ep in range(episodes):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Per-step logging for plotting: actions, middle-row tactile sensors, and per-step rewards
        actions_per_step = []
        mid_row_sensors_per_step = []  # list of (s3, s4, s5)
        rewards_per_step = []

        initial_obs = get_observation(data)
        initial_obs = np.nan_to_num(initial_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        if obs_rms is not None:
            # do NOT update obs_rms during validation; only normalize
            normalized = obs_rms.normalize(initial_obs)
            if not np.isfinite(normalized).all():
                print(f"Warning: non-finite normalized reset observation in validation episode {ep+1}; skipping episode.")
                continue
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1e6, neginf=-1e6)
            state_stacker.reset(normalized)
        else:
            state_stacker.reset(initial_obs)

        # Per-episode startup perturbation bookkeeping (mirror training behavior)
        # active_perturb_force is a 6-vector applied to data.xfrc_applied when active
        active_perturb_force = np.zeros(6, dtype=np.float32)
        # Resample perturbation according to PERTURB_RESAMPLE_EPISODES
        if ep % PERTURB_RESAMPLE_EPISODES == 0:
            if force_given is None:
                Fx_pf = np.random.uniform(-PERTURB_FORCE_SCALE, PERTURB_FORCE_SCALE)
            else:
                Fx_pf = force_given
            active_perturb_force[0] = Fx_pf
            # ensure no torque components are present (pure-x perturbation)
            active_perturb_force[3:] = 0.0

        active_perturb_steps_remaining = 0
        perturb_started = False
        episode_step = 0
        perturb_mask = []  # record whether perturbation was active at each step

        state = state_stacker.get_state()
        if not np.isfinite(state).all():
            print(f"Warning: non-finite stacked reset state in validation episode {ep+1}; skipping episode.")
            continue
        ep_reward = 0.0

        steps = 0
        done = False
        bad_episode = False
        while not done and steps < max_steps:
            allow_actions = episode_step >= ACTION_START_DELAY
            if allow_actions:
                # Select deterministic continuous action: use mean (actor output) and tanh-squash
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
            # Apply x-force directly to the body's spatial force vector.
            data.xfrc_applied[:] = 0
            Fx = np.asarray(action_scaled[0]).item()
            data.xfrc_applied[box_body_id][0] = Fx
            # Torque-from-height option removed; force is applied at the COM only.

            # Check for starting the delayed perturbation (same semantics as training)
            if (not perturb_started) and (episode_step >= PERTURB_START_DELAY):
                active_perturb_steps_remaining = PERTURB_APPLY_STEPS
                perturb_started = True

            # If perturbation is active, add it to the applied spatial force
            if active_perturb_steps_remaining > 0:
                data.xfrc_applied[box_body_id] = data.xfrc_applied[box_body_id] + active_perturb_force
                active_perturb_steps_remaining -= 1
                perturb_mask.append(True)
            else:
                perturb_mask.append(False)

            mujoco.mj_step(model, data)
            steps += 1
            episode_step += 1

            obs = get_observation(data)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

            # record action (force applied) and the middle-row three touch sensors
            try:
                # record the scaled action (force) applied in x
                actions_per_step.append(float(Fx))
            except Exception:
                # fallback: try to record from action_scaled
                try:
                    actions_per_step.append(float(action_scaled[0]))
                except Exception:
                    actions_per_step.append(0.0)
            # observation already holds the three middle-row sensors
            mid_row_sensors_per_step.append([float(obs[0]), float(obs[1]), float(obs[2])])

            pos = np.asarray(data.body('box_body').xpos, dtype=np.float32)
            out_of_bounds = (abs(pos[0]) > BOUNDARY_LIMIT) or (abs(pos[1]) > BOUNDARY_LIMIT)

            if obs_rms is not None:
                obs_norm = obs_rms.normalize(obs)
                if not np.isfinite(obs_norm).all():
                    print(f"Warning: non-finite normalized observation during validation episode {ep+1}; aborting episode.")
                    bad_episode = True
                    break
                obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                obs_norm = obs
            state_stacker.push(obs_norm)
            state = state_stacker.get_state()
            if not np.isfinite(state).all():
                print(f"Warning: non-finite stacked state during validation episode {ep+1}; aborting episode.")
                bad_episode = True
                break

            reward, alive = compute_reward(data, Fx, TILT_LIMIT_RADIANS, out_of_bounds=out_of_bounds)

            timeout = (steps >= max_steps)
            done = (not alive) or timeout

            ep_reward += reward
            # record per-step reward for plotting
            try:
                rewards_per_step.append(float(reward))
            except Exception:
                rewards_per_step.append(0.0)

            if render and viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.01)

        if bad_episode:
            print(f"Skipping validation episode {ep+1} due to non-finite state.")
            continue

        rewards.append(ep_reward)
        print(f"Validation episode {ep+1}/{episodes} reward: {ep_reward:.2f} steps: {steps}")
        # After each episode, save a plot of action vs steps and middle-row touch sensors
        try:
                if len(actions_per_step) > 0:
                    xs = np.arange(1, len(actions_per_step) + 1)
                    actions_arr = np.array(actions_per_step, dtype=np.float32)
                    mid = np.array(mid_row_sensors_per_step, dtype=np.float32)
                    rewards_arr = np.array(rewards_per_step, dtype=np.float32)
                    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

                    # Plot action as a continuous polyline but color segments by sign
                    eps = 1e-6
                    signs = np.sign(actions_arr)
                    signs[np.abs(actions_arr) <= eps] = 0
                    # find segment boundaries where sign changes
                    boundaries = np.flatnonzero(np.concatenate(([0], signs[1:] != signs[:-1], [1])))
                    start = 0
                    labels_added = set()
                    for b in boundaries:
                        end = b
                        seg_x = xs[start:end]
                        seg_y = actions_arr[start:end]
                        if seg_x.size > 0:
                            seg_sign = np.sign(seg_y).mean()
                            if seg_sign > 0:
                                color = 'tab:blue'
                                label = 'action > 0'
                            elif seg_sign < 0:
                                color = 'tab:red'
                                label = 'action < 0'
                            else:
                                color = 'gray'
                                label = 'action = 0'
                            axs[0].plot(seg_x, seg_y, color=color, linewidth=1.5)
                            if label not in labels_added:
                                axs[0].plot([], [], color=color, label=label)
                                labels_added.add(label)
                        start = end
                    # horizontal zero line
                    axs[0].axhline(0.0, color='k', linewidth=0.8, linestyle='--')
                    axs[0].set_ylabel('Action (force)')
                    # Shade startup perturbation windows (if any)
                    try:
                        if len(perturb_mask) > 0:
                            pm = np.array(perturb_mask, dtype=bool)
                            in_span = False
                            start_idx = 0
                            for i, val in enumerate(pm):
                                if val and not in_span:
                                    in_span = True
                                    start_idx = i
                                elif (not val) and in_span:
                                    in_span = False
                                    start_line = start_idx + 1 - 0.5
                                    end_line = i + 0.5
                                    axs[0].axvline(start_line, color='black', linestyle='--', linewidth=1.0)
                                    axs[0].axvline(end_line, color='black', linestyle='--', linewidth=1.0)
                            if in_span:
                                start_line = start_idx + 1 - 0.5
                                end_line = len(pm) + 0.5
                                axs[0].axvline(start_line, color='black', linestyle='--', linewidth=1.0)
                                axs[0].axvline(end_line, color='black', linestyle='--', linewidth=1.0)
                            # Add legend entry for perturbation
                            handles, labels = axs[0].get_legend_handles_labels()
                            line_handle = Line2D([0], [0], color='black', linestyle='--', linewidth=1.0, label='startup perturb')
                            handles.append(line_handle)
                            axs[0].legend(handles=handles)
                        else:
                            axs[0].legend()
                    except Exception:
                        axs[0].legend()

                    axs[1].plot(xs, mid[:, 0], label='sensor_p1_0')
                    axs[1].plot(xs, mid[:, 1], label='sensor_0_0')
                    axs[1].plot(xs, mid[:, 2], label='sensor_m1_0')
                    axs[1].set_ylabel('Touch sensor z')
                    axs[1].legend()

                    # Plot per-step rewards on a separate subplot
                    axs[2].plot(xs, rewards_arr, color='tab:green', linewidth=1.2)
                    axs[2].set_xlabel('Step')
                    axs[2].set_ylabel('Reward')

                fig.suptitle(f'Validation Episode {ep+1}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                if file_prefix:
                    filename = f"{file_prefix}_ep_{ep+1}.png"
                else:
                    filename = f"validation_ep_{ep+1}.png"
                fname = os.path.join(save_dir, filename)
                fig.savefig(fname)
                plt.close(fig)
                saved_figs.append(fname)
        except Exception as e:
            print(f"Warning: failed to save validation plot: {e}")

    if viewer is not None:
        viewer.close()

    avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else float("nan")
    print(f"Validation completed. Average reward over {episodes} episodes: {avg_reward:.2f}")
    return rewards, saved_figs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pretrained/latest/box_ppo_model_best.pt', help='Path to model weights')
    parser.add_argument('--obs_rms', type=str, default='pretrained/latest/obs_rms_best.npz', help='Path to obs_rms .npz file')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default=None, help='Optional directory to store validation artifacts')
    parser.add_argument('--file_prefix', type=str, default=None, help='Optional filename prefix for saved plots')
    parser.add_argument('--force_given', type=float, default=None, help='Optional fixed startup perturbation force value')
    args = parser.parse_args()

    rewards, paths = run_validation(
        args.model,
        args.obs_rms,
        episodes=args.episodes,
        render=args.render,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        file_prefix=args.file_prefix,
        force_given=args.force_given,
    )
    print(f"Saved plots: {paths}")
