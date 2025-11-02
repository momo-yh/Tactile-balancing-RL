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
import time
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
FORCES = [1, 5, 10, 20, 30, 40, 50, 70, 80, 90, 100, 150, 200]  # forces in +x direction to test
# Number of simulation steps the force is actively applied (1 as requested)
APPLIED_STEPS = 1
# Number of steps to record after the applied impulse
RECORD_STEPS = 300
# Begin applying the force at this (1-based) step index after reset
APPLY_AT_STEP = 150
MODEL_PATH = "model.xml"
SAVE_DIR = os.path.join("validation", "force_compare")
os.makedirs(SAVE_DIR, exist_ok=True)


def run_test():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    box_body_id = data.body('box_body').id

    # store per-force per-step mid-row sensor readings (steps x 3)
    all_mid = {}  # force -> array shape (STEPS_PER_FORCE, 3)

    for f in FORCES:
        print(f"Running force {f}: apply {APPLIED_STEPS} step(s), then record {RECORD_STEPS} steps...")
        # Reset sim before applying this force
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # We'll record the pre-impulse steps, the applied step(s), and the post-impulse RECORD_STEPS
        pre_steps = max(0, APPLY_AT_STEP - 1)
        total_steps = pre_steps + APPLIED_STEPS + RECORD_STEPS

        mid_vals = []
        for t in range(1, total_steps + 1):
            # apply force only during the APPLIED_STEPS starting at APPLY_AT_STEP
            if (t >= APPLY_AT_STEP) and (t < APPLY_AT_STEP + APPLIED_STEPS):
                data.xfrc_applied[:] = 0
                data.xfrc_applied[box_body_id][0] = float(f)
            else:
                data.xfrc_applied[:] = 0

            mujoco.mj_step(model, data)

            obs = get_observation(data)
            # observation already contains only the three middle-row sensors
            mid = [float(obs[0]), float(obs[1]), float(obs[2])]
            mid_vals.append(mid)

        # convert to array and store for this force
        mid_arr = np.array(mid_vals, dtype=np.float32)  # shape (RECORD_STEPS, 3)
        all_mid[f] = mid_arr
    # Compute and print/save maxima for each force (each plotted line)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    print('\nMaximum values for each force and middle-row sensor:')
    sensor_names = ['sensor_p1_0', 'sensor_0_0', 'sensor_m1_0']
    # Print header
    print(','.join(['force'] + sensor_names))
    for f in FORCES:
        arr = all_mid[f]
        maxs = arr.max(axis=0)
        print(f"{f},{maxs[0]:.6f},{maxs[1]:.6f},{maxs[2]:.6f}")

    # Save maxima to CSV for later inspection
    csv_path = os.path.join(SAVE_DIR, f'force_midrow_max_{ts}.csv')
    with open(csv_path, 'w') as fh:
        fh.write(','.join(['force'] + sensor_names) + '\n')
        for f in FORCES:
            m = all_mid[f].max(axis=0)
            fh.write(f"{f},{m[0]},{m[1]},{m[2]}\n")
    print(f"Saved maxima CSV to: {csv_path}\n")

    # Plot all traces on the same figure
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    fname = os.path.join(SAVE_DIR, f'force_midrow_compare_{ts}.png')

    plt.figure(figsize=(12, 6))
    pre_steps = max(0, APPLY_AT_STEP - 1)
    total_steps = pre_steps + APPLIED_STEPS + RECORD_STEPS
    xs = np.arange(1, total_steps + 1)
    # Names for the three middle-row sensors (based on get_observation ordering)
    sensor_names = ['sensor_p1_0', 'sensor_0_0', 'sensor_m1_0']

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i in range(3):
        ax = axs[i]
        for f in FORCES:
            ax.plot(xs, all_mid[f][:, i], label=f"force={f}")
        ax.set_ylabel(sensor_names[i])
        ax.grid(True)
        ax.legend(loc='upper right')

    axs[-1].set_xlabel('Step')
    fig.suptitle(f'Middle-row sensor response: {APPLIED_STEPS}-step impulse at step {APPLY_AT_STEP} (recorded {total_steps} steps)')
    # annotate impulse step on each subplot
    for ax in axs:
        ax.axvline(APPLY_AT_STEP, color='k', linestyle='--', linewidth=0.8, label='impulse')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved comparison plot to: {fname}")


if __name__ == '__main__':
    run_test()
