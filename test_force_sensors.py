"""test_force_sensors.py

Apply constant +x forces (list of magnitudes) for a fixed number of steps
and record the middle-row tactile sensors. For each force we reset the sim,
apply the force for N steps, record the 3 mid-row sensors per step, then
plot the average mid-row sensor value over time for each force on a single
comparison figure.

Usage:
    python test_force_sensors.py

Requires mujoco Python bindings available in the environment.
"""

import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from features import build_observation

try:
    import mujoco
except Exception as e:
    print(f"Error: could not import mujoco: {e}")
    print("This script requires MuJoCo Python bindings. Exiting.")
    raise

def get_observation(data):
    return build_observation(data)

# Configuration
FORCES = [200]  # forces in +x direction to test
# Number of simulation steps the force is actively applied (1 as requested)
APPLIED_STEPS = 1
# Number of steps to record after the applied impulse
RECORD_STEPS = 1000
# Begin applying the force at this (1-based) step index after reset
APPLY_AT_STEP = 150
MODEL_PATH = "model.xml"
SAVE_DIR = os.path.join("validation", "force_compare")
os.makedirs(SAVE_DIR, exist_ok=True)


def run_test():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    box_body_id = data.body('box_body').id

    # store per-force per-step observation features (tilt diff, acceleration sign)
    all_obs = {}  # force -> array shape (STEPS_PER_FORCE, 2)
    # store per-force per-step x-position of the box body (steps,)
    all_xpos = {}  # force -> array shape (STEPS_PER_FORCE,)

    for f in FORCES:
        print(f"Running force {f}: apply {APPLIED_STEPS} step(s), then record {RECORD_STEPS} steps...")
        # Reset sim before applying this force
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # We'll record the pre-impulse steps, the applied step(s), and the post-impulse RECORD_STEPS
        pre_steps = max(0, APPLY_AT_STEP - 1)
        total_steps = pre_steps + APPLIED_STEPS + RECORD_STEPS

        obs_vals = []
        xpos_vals = []
        for t in range(1, total_steps + 1):
            # apply force only during the APPLIED_STEPS starting at APPLY_AT_STEP
            if (t >= APPLY_AT_STEP) and (t < APPLY_AT_STEP + APPLIED_STEPS):
                data.xfrc_applied[:] = 0
                data.xfrc_applied[box_body_id][0] = float(f)
            else:
                data.xfrc_applied[:] = 0

            mujoco.mj_step(model, data)

            # record observation feature values (tilt difference and acceleration sign)
            obs = get_observation(data)
            if obs.size >= 2:
                obs_vals.append([float(obs[0]), float(obs[1])])
            else:
                obs_vals.append([0.0, 0.0])
            # record bottom-right site x-position (world frame)
            try:
                site_x = float(data.site('site_p1_m1').xpos[0])
            except Exception:
                # fallback to body xpos if site not found
                try:
                    site_x = float(data.body('box_body').xpos[0])
                except Exception:
                    site_x = 0.0
            xpos_vals.append(site_x)

        # convert to array and store for this force
        obs_arr = np.array(obs_vals, dtype=np.float32)  # shape (STEPS, 2)
        xpos_arr = np.array(xpos_vals, dtype=np.float32)  # shape (STEPS,)
        # calibrate baseline using pre-impulse steps (use mean of pre_steps)
        if pre_steps > 0:
            baseline = float(xpos_arr[:pre_steps].mean())
        else:
            baseline = float(xpos_arr[0]) if xpos_arr.size > 0 else 0.0
        # store displacement relative to baseline
        disp_arr = xpos_arr - baseline
        all_obs[f] = obs_arr
        all_xpos[f] = disp_arr
    # Compute and print/save maxima for each force (each plotted line)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    print('\nMin/Max values for each force observation feature (and final x-position):')
    feature_names = ['tilt_diff', 'ang_vel_direction']
    # Print header
    print(','.join(['force'] + [f"{name}_min" for name in feature_names] + [f"{name}_max" for name in feature_names]))
    for f in FORCES:
        arr = all_obs[f]
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        print(f"{f},{mins[0]:.6f},{mins[1]:.6f},{maxs[0]:.6f},{maxs[1]:.6f}")

    # Save maxima to CSV for later inspection (also include final x-position)
    csv_path = os.path.join(SAVE_DIR, f'force_observation_extrema_{ts}.csv')
    with open(csv_path, 'w') as fh:
        header = ['force']
        header += [f"{name}_min" for name in feature_names]
        header += [f"{name}_max" for name in feature_names]
        header.append('xpos_final')
        fh.write(','.join(header) + '\n')
        for f in FORCES:
            arr = all_obs[f]
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            xpos_final = float(all_xpos[f][-1]) if (f in all_xpos and all_xpos[f].size > 0) else 0.0
            fh.write(
                f"{f},{mins[0]},{mins[1]},{maxs[0]},{maxs[1]},{xpos_final}\n"
            )
    print(f"Saved maxima CSV to: {csv_path}\n")

    # Plot all traces on the same figure
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    fname = os.path.join(SAVE_DIR, f'force_observation_compare_{ts}.png')

    pre_steps = max(0, APPLY_AT_STEP - 1)
    total_steps = pre_steps + APPLIED_STEPS + RECORD_STEPS
    xs = np.arange(1, total_steps + 1)

    # Names for observation features
    feature_labels = ['tilt_diff', 'ang_vel_direction']

    # Create 3 subplots: two observation features + one for box x-position
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i in range(2):
        ax = axs[i]
        for f in FORCES:
            ax.plot(xs, all_obs[f][:, i], label=f"force={f}")
        ax.set_ylabel(feature_labels[i])
        ax.grid(True)
        ax.legend(loc='upper right')

    # Third subplot: box x-position
    ax = axs[2]
    for f in FORCES:
        if f in all_xpos:
            ax.plot(xs, all_xpos[f], label=f"force={f}")
    ax.set_ylabel('box_xpos (m)')
    ax.set_xlabel('Step')
    ax.grid(True)
    ax.legend(loc='upper right')

    fig.suptitle(f'Observation features + box x-position: {APPLIED_STEPS}-step impulse at step {APPLY_AT_STEP} (recorded {total_steps} steps)')
    # annotate impulse step on each subplot
    for a in axs:
        a.axvline(APPLY_AT_STEP, color='k', linestyle='--', linewidth=0.8, label='impulse')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved comparison plot to: {fname}")


if __name__ == '__main__':
    run_test()
