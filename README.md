# Balancing Ball PPO Agent

This repository contains a MuJoCo-based balancing task where a neural policy learned with Proximal Policy Optimisation (PPO) keeps a box centred on top of a ball. Classical controllers have been replaced entirely by a continuous Gaussian actor; the agent learns from tactile feedback gathered at the base of the box.

## System Overview

| Component | Description |
|-----------|-------------|
| **Observation stack** | Three middle-row tactile sensors (`sensor_p1_0`, `sensor_0_0`, `sensor_m1_0`) stacked over `STACK_SIZE` frames. The stack is optionally normalised online by `RunningMeanStd` (`USE_OBS_NORMALIZATION`). |
| **Policy & value nets** | Shared `ActorCritic` module with two-layer tanh MLPs for the mean action and state value. The actor keeps a learnable per-action `log_std` and samples in the unconstrained space before squashing with `tanh`. |
| **Action space** | Single continuous force along the world x-axis, later scaled by `ACTION_FORCE_MAGNITUDE` before applying it to the MuJoCo body. |
| **Rollouts** | A `RolloutBuffer` collects on-policy trajectories, computes GAE advantages, and feeds mini-batches into PPO updates. |

### Reward shaping (`reward.py`)

- **Alive bonus**: `ALIVE_BONUS` (default `+10`) is granted only when all three tactile sensors report contact (`|sensor| > SENSOR_CONTACT_TOLERANCE`) and the tilt stays below `±TILT_LIMIT_DEGREES`.
- **Smoothness penalties**: `-ANGULAR_VEL_PENALTY * tilt_rate²` discourages oscillations, while `-ACTION_ENERGY_PENALTY * force²` suppresses violent pushes.
- **Sensor agreement bonus**: When the three sensors agree, a graded bonus up to `SENSOR_MATCH_BONUS` (default `10`) is added. The bonus scales linearly with the spread (`max(sensor) - min(sensor)`); perfect agreement (`spread = 0`) yields the full reward, and the contribution tapers to zero once the spread exceeds `SENSOR_MATCH_TOLERANCE`.
- **Failure penalty**: `-FALL_PENALTY` (default `-100`) triggers when the tilt or position exits the safe region.

Episodes terminate immediately on failure or after the configured step horizon (`MAX_EPISODE_STEPS`).

## Training pipeline

```cmd
python train_box_ppo.py
python train_box_ppo.py --resume-dir <folder> --resume-tag {best,last}
```

Default hyper-parameters (adjustable through environment variables and `settings.py`):

- `ROLLOUT_LENGTH=1024`, `BATCH_SIZE=32`, `PPO_EPOCHS=32`, `LR=1e-4`.
- Startup perturbations (`PERTURB_*`) periodically nudge the system to test robustness.
- Observation statistics and checkpoints are written to `pretrained/<timestamp>_TILTXX_RL/`, with `pretrained/latest/` mirroring the newest artefacts.

### Learning-rate scheduling

`train_box_ppo.py` now supports late-stage learning-rate decay via `torch.optim.lr_scheduler.MultiStepLR`:

| Environment variable | Default | Effect |
|----------------------|---------|--------|
| `LR_DECAY_FRACTION`  | `0.75`  | Fraction of expected PPO updates completed before the first decay. |
| `LR_DECAY_GAMMA`     | `0.1`   | Multiplicative drop applied to the optimiser’s LR (set `1.0` to disable). |

The trainer estimates the total number of PPO updates (`MAX_TRAINING_STEPS // ROLLOUT_LENGTH`), schedules a single decay milestone, and logs the drop when it fires. This lets the policy take smaller “refinement” steps near convergence (e.g. `1e-4 → 1e-5`).

## Evaluation & analysis

### Deterministic rollouts

```cmd
python validate_best.py \
	--model pretrained/latest/box_ppo_model_best.pt \
	--obs_rms pretrained/latest/obs_rms_best.npz \
	--episodes 5
```

`validate_best.py` replays the policy deterministically, logs per-step rewards/actions, and can optionally render (`--render`) if a MuJoCo viewer is available. Validation artefacts are stored under `validation/<timestamp>/`.

### Controlled vs open-loop comparison

```cmd
python validate_compare.py --model pretrained/latest/box_ppo_model_best.pt --episodes 3
```

For each episode, `validate_compare.py` clones the initial MuJoCo state and runs the simulation twice: once with the learned controller, once with zero actions. It plots bottom-row tactile forces in a two-tier figure so you can inspect how the controller stabilises contact pressure relative to the passive baseline. Average rewards for both modes are reported at the end of the run.

### Additional utilities

- `test_force_sensors.py` fires diagnostic impulses to characterise tactile responses.


## Requirements

- Python 3.11+
- MuJoCo 3.x Python bindings (with a valid MuJoCo installation and assets)
- PyTorch (CUDA optional but recommended)
- NumPy, Matplotlib


## Tips

- Enable observation normalisation with `USE_OBS_NORMALIZATION=1` for better training stability.
- Use `SHORT_TEST=1` to run a quick 20k-step smoke training pass before committing to a full session.
- Check `pretrained/latest/` for the most recent `box_ppo_model_{best,last}.pt` weights and matching `obs_rms_*.npz` files before launching a validation job.
