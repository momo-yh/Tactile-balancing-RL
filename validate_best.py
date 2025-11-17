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

from train_ppo_mjx import ActorCritic, StateStacker, RunningMeanStd, get_observation
from features import mid_row_sensors
from reward import compute_reward
from settings import (
    STACK_SIZE, OBS_DIM, STATE_DIM, ACTION_DIM, HIDDEN_DIM,
    ACTION_FORCE_MAGNITUDE, PERTURB_FORCE_SCALE, PERTURB_APPLY_STEPS,
    PERTURB_START_DELAY, PERTURB_RESAMPLE_EPISODES, ACTION_START_DELAY, BOUNDARY_LIMIT,
    TILT_LIMIT_RADIANS, USE_OBS_NORMALIZATION,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_validation(
    model_path,
    obs_rms_path=None,
    episodes=10,
    render=False,
    max_steps=1000,
    save_dir=None,
    file_prefix=None,
    force_given=200,
):
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    box_body_id = data.body('box_body').id

    net = ActorCritic(STATE_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

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
            importlib.import_module("mujoco.viewer")
        except Exception as e:
            print(f"Warning: could not import mujoco.viewer: {e}")
            render = False
        else:
            viewer = mujoco.viewer.launch_passive(model, data)

    if save_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join("validation", ts)
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(episodes):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        actions_per_step = []
        obs_features_per_step = []
        mid_row_sensors_per_step = []
        rewards_per_step = []

        initial_obs = get_observation(data)
        initial_obs = np.nan_to_num(initial_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        if obs_rms is not None:
            normalized = obs_rms.normalize(initial_obs)
            if not np.isfinite(normalized).all():
                print(f"Warning: non-finite normalized reset observation in validation episode {ep + 1}; skipping episode.")
                continue
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1e6, neginf=-1e6)
            state_stacker.reset(normalized)
        else:
            state_stacker.reset(initial_obs)

        active_perturb_force = np.zeros(6, dtype=np.float32)
        if ep % PERTURB_RESAMPLE_EPISODES == 0:
            if force_given is None:
                Fx_pf = np.random.uniform(-PERTURB_FORCE_SCALE, PERTURB_FORCE_SCALE)
            else:
                Fx_pf = force_given
            active_perturb_force[0] = Fx_pf
            active_perturb_force[3:] = 0.0

        active_perturb_steps_remaining = 0
        perturb_started = False
        episode_step = 0
        perturb_mask = []

        state = state_stacker.get_state()
        if not np.isfinite(state).all():
            print(f"Warning: non-finite stacked reset state in validation episode {ep + 1}; skipping episode.")
            continue

        ep_reward = 0.0
        steps = 0
        done = False
        bad_episode = False

        while not done and steps < max_steps:
            allow_actions = episode_step >= ACTION_START_DELAY
            if allow_actions:
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
            data.xfrc_applied[:] = 0
            Fx = np.asarray(action_scaled[0]).item()
            data.xfrc_applied[box_body_id][0] = Fx

            if (not perturb_started) and (episode_step >= PERTURB_START_DELAY):
                active_perturb_steps_remaining = PERTURB_APPLY_STEPS
                perturb_started = True

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
            obs_features_per_step.append([
                float(obs[0]) if obs.size > 0 else 0.0,
                float(obs[1]) if obs.size > 1 else 0.0,
            ])

            try:
                actions_per_step.append(float(Fx))
            except Exception:
                try:
                    actions_per_step.append(float(action_scaled[0]))
                except Exception:
                    actions_per_step.append(0.0)

            mid_contacts = mid_row_sensors(data)
            mid_row_sensors_per_step.append([
                float(mid_contacts[0]),
                float(mid_contacts[1]),
                float(mid_contacts[2]),
            ])

            pos = np.asarray(data.body('box_body').xpos, dtype=np.float32)
            out_of_bounds = (abs(pos[0]) > BOUNDARY_LIMIT) or (abs(pos[1]) > BOUNDARY_LIMIT)

            if obs_rms is not None:
                obs_norm = obs_rms.normalize(obs)
                if not np.isfinite(obs_norm).all():
                    print(f"Warning: non-finite normalized observation during validation episode {ep + 1}; aborting episode.")
                    bad_episode = True
                    break
                obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                obs_norm = obs
            state_stacker.push(obs_norm)
            state = state_stacker.get_state()
            if not np.isfinite(state).all():
                print(f"Warning: non-finite stacked state during validation episode {ep + 1}; aborting episode.")
                bad_episode = True
                break

            reward, alive = compute_reward(data, Fx, TILT_LIMIT_RADIANS, out_of_bounds=out_of_bounds)
            timeout = (steps >= max_steps)
            done = (not alive) or timeout

            ep_reward += reward
            try:
                rewards_per_step.append(float(reward))
            except Exception:
                rewards_per_step.append(0.0)

            if render and viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.01)

        if bad_episode:
            print(f"Skipping validation episode {ep + 1} due to non-finite state.")
            continue

        rewards.append(ep_reward)
        print(f"Validation episode {ep + 1}/{episodes} reward: {ep_reward:.2f} steps: {steps}")

        try:
            if len(actions_per_step) > 0:
                xs = np.arange(1, len(actions_per_step) + 1)
                actions_arr = np.array(actions_per_step, dtype=np.float32)
                obs_feats = np.array(obs_features_per_step, dtype=np.float32)
                mid = np.array(mid_row_sensors_per_step, dtype=np.float32)
                rewards_arr = np.array(rewards_per_step, dtype=np.float32)

                fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                ax_action, ax_obs, ax_mid, ax_reward = axs

                eps = 1e-6
                signs = np.sign(actions_arr)
                signs[np.abs(actions_arr) <= eps] = 0
                boundaries = np.flatnonzero(np.concatenate(([0], signs[1:] != signs[:-1], [1])))
                start_idx = 0
                labels_added = set()
                for boundary in boundaries:
                    end_idx = boundary
                    seg_x = xs[start_idx:end_idx]
                    seg_y = actions_arr[start_idx:end_idx]
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
                        ax_action.plot(seg_x, seg_y, color=color, linewidth=1.5)
                        if label not in labels_added:
                            ax_action.plot([], [], color=color, label=label)
                            labels_added.add(label)
                    start_idx = end_idx

                ax_action.axhline(0.0, color='k', linewidth=0.8, linestyle='--')
                ax_action.set_ylabel('Action (force)')

                try:
                    if len(perturb_mask) > 0:
                        pm = np.array(perturb_mask, dtype=bool)
                        in_span = False
                        span_start = 0
                        for i, active in enumerate(pm):
                            if active and not in_span:
                                in_span = True
                                span_start = i
                            elif (not active) and in_span:
                                in_span = False
                                ax_action.axvline(span_start + 1 - 0.5, color='black', linestyle='--', linewidth=1.0)
                                ax_action.axvline(i + 0.5, color='black', linestyle='--', linewidth=1.0)
                        if in_span:
                            ax_action.axvline(span_start + 1 - 0.5, color='black', linestyle='--', linewidth=1.0)
                            ax_action.axvline(len(pm) + 0.5, color='black', linestyle='--', linewidth=1.0)
                        handles, labels = ax_action.get_legend_handles_labels()
                        line_handle = Line2D([0], [0], color='black', linestyle='--', linewidth=1.0, label='startup perturb')
                        handles.append(line_handle)
                        ax_action.legend(handles=handles)
                    else:
                        ax_action.legend()
                except Exception:
                    ax_action.legend()

                if obs_feats.size > 0:
                    ax_obs.plot(xs, obs_feats[:, 0], color='tab:purple', label='tilt_diff (p1_0 - m1_0)')
                    ax_obs.step(xs, obs_feats[:, 1], where='post', color='tab:orange', linewidth=1.2, label='ang_vel_direction')
                    ax_obs.set_ylabel('Observation features')
                    ax_obs.legend()
                    ax_obs.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

                ax_mid.plot(xs, mid[:, 0], label='sensor_p1_0')
                ax_mid.plot(xs, mid[:, 1], label='sensor_0_0')
                ax_mid.plot(xs, mid[:, 2], label='sensor_m1_0')
                ax_mid.set_ylabel('Touch sensor z')
                ax_mid.legend()
                ax_mid.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

                ax_reward.plot(xs, rewards_arr, color='tab:green', linewidth=1.2)
                ax_reward.set_xlabel('Step')
                ax_reward.set_ylabel('Reward')
                ax_reward.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

                ax_action.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

                fig.suptitle(f'Validation Episode {ep + 1}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                if file_prefix:
                    filename = f"{file_prefix}_ep_{ep + 1}.png"
                else:
                    filename = f"validation_ep_{ep + 1}.png"
                fname = os.path.join(save_dir, filename)
                fig.savefig(fname)
                plt.close(fig)
                saved_figs.append(fname)
        except Exception as e:
            print(f"Warning: failed to save validation plot: {e}")

    if viewer is not None:
        viewer.close()

    avg_reward = float(np.mean(rewards)) if rewards else float("nan")
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
