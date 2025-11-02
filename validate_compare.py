import mujoco
import numpy as np
import os
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import argparse, importlib
from train_box_ppo import ActorCritic, StateStacker, RunningMeanStd, get_observation
from features import tactile_sensors, SENSOR_NAMES

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

    rewards_controlled = []
    rewards_open = []
    saved_figs = []

    bottom_indices = [6, 7, 8]
    bottom_labels = [SENSOR_NAMES[i] for i in bottom_indices]

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

    last_perturb_force = np.zeros(6, dtype=np.float32)

    def simulate_episode(apply_control, base_qpos, base_qvel, perturb_force):
        stacker = StateStacker(OBS_DIM, STACK_SIZE)

        data.qpos[:] = base_qpos
        data.qvel[:] = base_qvel
        data.xfrc_applied[:] = 0
        mujoco.mj_forward(model, data)

        initial_obs = get_observation(data)
        initial_obs = np.nan_to_num(initial_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        if obs_rms is not None:
            normalized = obs_rms.normalize(initial_obs)
            if not np.isfinite(normalized).all():
                return {"bad": True}
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1e6, neginf=-1e6)
            stacker.reset(normalized)
        else:
            stacker.reset(initial_obs)

        state = stacker.get_state()
        if not np.isfinite(state).all():
            return {"bad": True}

        bottom_history = []
        reward_history = []
        total_reward = 0.0

        active_perturb_steps_remaining = 0
        perturb_started = False
        episode_step = 0
        steps = 0

        while steps < max_steps:
            allow_actions = episode_step >= ACTION_START_DELAY
            if apply_control and allow_actions:
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
            Fx = float(action_scaled[0]) if action_scaled.size > 0 else 0.0
            data.xfrc_applied[box_body_id][0] = Fx

            if (not perturb_started) and (episode_step >= PERTURB_START_DELAY):
                active_perturb_steps_remaining = PERTURB_APPLY_STEPS
                perturb_started = True

            if active_perturb_steps_remaining > 0:
                data.xfrc_applied[box_body_id] = data.xfrc_applied[box_body_id] + perturb_force
                active_perturb_steps_remaining -= 1

            mujoco.mj_step(model, data)
            steps += 1
            episode_step += 1

            sensors_full = tactile_sensors(data)
            sensors_full = np.nan_to_num(sensors_full, nan=0.0, posinf=0.0, neginf=0.0)
            bottom_history.append(sensors_full[bottom_indices])

            pos = np.asarray(data.body('box_body').xpos, dtype=np.float32)
            out_of_bounds = (abs(pos[0]) > BOUNDARY_LIMIT) or (abs(pos[1]) > BOUNDARY_LIMIT)

            obs = get_observation(data)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            if obs_rms is not None:
                obs_norm = obs_rms.normalize(obs)
                if not np.isfinite(obs_norm).all():
                    return {"bad": True}
                obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                obs_norm = obs
            stacker.push(obs_norm)
            state = stacker.get_state()
            if not np.isfinite(state).all():
                return {"bad": True}

            reward, alive = compute_reward(data, Fx, TILT_LIMIT_RADIANS, out_of_bounds=out_of_bounds)
            total_reward += reward
            reward_history.append(reward)

            if render and viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.01)

            if not alive:
                break

        return {
            "bad": False,
            "bottom_sensors": np.vstack(bottom_history) if bottom_history else np.zeros((0, len(bottom_indices)), dtype=np.float32),
            "rewards": np.array(reward_history, dtype=np.float32),
            "total_reward": float(total_reward),
            "steps": steps,
        }

    for ep in range(episodes):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        base_qpos = data.qpos.copy()
        base_qvel = data.qvel.copy()

        if ep % PERTURB_RESAMPLE_EPISODES == 0:
            last_perturb_force = np.zeros(6, dtype=np.float32)
            Fx_pf = np.random.uniform(-PERTURB_FORCE_SCALE, PERTURB_FORCE_SCALE)
            Fx_pf = 100  ##TODO
            last_perturb_force[0] = Fx_pf
            last_perturb_force[3:] = 0.0

        results = {}
        episode_bad = False
        for label, control_flag in (("with_control", True), ("no_control", False)):
            res = simulate_episode(control_flag, base_qpos, base_qvel, last_perturb_force)
            if res.get("bad", False):
                print(f"Skipping validation episode {ep+1} ({label}) due to non-finite state.")
                episode_bad = True
                break
            results[label] = res

        if episode_bad:
            continue

        rewards_controlled.append(results["with_control"]["total_reward"])
        rewards_open.append(results["no_control"]["total_reward"])

        print(
            f"Episode {ep+1}/{episodes} | reward(control)={results['with_control']['total_reward']:.2f} "
            f"reward(no-control)={results['no_control']['total_reward']:.2f}"
        )

        try:
            fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

            time_control = np.arange(1, results["with_control"]["bottom_sensors"].shape[0] + 1)
            for idx, label in enumerate(bottom_labels):
                axs[0].plot(time_control, results["with_control"]["bottom_sensors"][:, idx], label=label)
            axs[0].set_title("Bottom sensors • With control")
            axs[0].set_ylabel('Force (z)')
            axs[0].legend(loc='upper right')

            time_open = np.arange(1, results["no_control"]["bottom_sensors"].shape[0] + 1)
            for idx, label in enumerate(bottom_labels):
                axs[1].plot(time_open, results["no_control"]["bottom_sensors"][:, idx], label=label)
            axs[1].set_title("Bottom sensors • No control")
            axs[1].set_xlabel('Step')
            axs[1].set_ylabel('Force (z)')
            axs[1].legend(loc='upper right')

            fig.suptitle(f'Bottom sensor comparison (episode {ep+1})')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            if file_prefix:
                filename = f"{file_prefix}_ep_{ep+1}.png"
            else:
                filename = f"compare_ep_{ep+1}.png"
            fname = os.path.join(save_dir, filename)
            fig.savefig(fname)
            plt.close(fig)
            saved_figs.append(fname)
        except Exception as e:
            print(f"Warning: failed to save validation comparison plot: {e}")

    if viewer is not None:
        viewer.close()

    avg_control = float(np.mean(rewards_controlled)) if rewards_controlled else float("nan")
    avg_open = float(np.mean(rewards_open)) if rewards_open else float("nan")
    print(
        f"Validation completed. Average reward with control: {avg_control:.2f} | "
        f"no control: {avg_open:.2f}"
    )
    return {"with_control": rewards_controlled, "no_control": rewards_open}, saved_figs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pretrained/latest/box_ppo_model_best.pt', help='Path to model weights')
    parser.add_argument('--obs_rms', type=str, default='pretrained/latest/obs_rms_best.npz', help='Path to obs_rms .npz file')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default=None, help='Optional directory to store validation artifacts')
    parser.add_argument('--file_prefix', type=str, default=None, help='Optional filename prefix for saved plots')
    args = parser.parse_args()

    rewards, paths = run_validation(
        args.model,
        args.obs_rms,
        episodes=args.episodes,
        render=args.render,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        file_prefix=args.file_prefix,
    )
    print(f"Saved plots: {paths}")
