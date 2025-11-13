# Balancing PPO Agent

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
# Balancing PPO Agent — Environment and Training Guide

This document explains the RL setup end to end: observation/action shapes, reward design, training loop and hyperparameters, runtime commands, common environment variables, how to inspect actual minibatches, and practical debugging tips.

## 1) Overview

- Task: learn a continuous control policy that keeps a box balanced on a ball in MuJoCo.
- Algorithm: standard on‑policy PPO with a shared Actor‑Critic network; the actor samples from a Gaussian and squashes with tanh to [-1, 1].
- Execution: single‑environment training loop (rollouts collected step‑by‑step in one simulator instance).

## 2) Requirements

- Python 3.11+
- MuJoCo Python bindings (with MuJoCo runtime/assets properly installed)
- PyTorch (CUDA recommended)
- NumPy, Matplotlib

## 3) Key files

- `train_box_ppo.py`: main training script (single environment PPO).
- `reward.py`: reward function and shaping constants (`ALIVE_BONUS`, `FALL_PENALTY`, `ACTION_ENERGY_PENALTY`, etc.).
- `settings.py`: shared dimensions and defaults (`STACK_SIZE`, `OBS_DIM`, `ACTION_DIM`, …).
- `validate_best.py` and `validate_compare.py`: evaluation scripts for deterministic rollout and controller vs open‑loop comparison.

## 4) Observations and actions
 # Balancing PPO Agent

This repository implements a MuJoCo balancing task where a learned PPO policy applies continuous forces to keep a box balanced on a ball. The agent relies on tactile sensors (three middle-row sensors) and learns a continuous Gaussian policy (actor) with a value head (critic).

This README was updated to reflect recent enhancements in the training scripts: curriculum-based startup perturbations, automatic validation hooks, and improved resume provenance (see "Recent changes" below).

## Quick summary

- Training scripts: `train_box_ppo.py` (classic single-env PPO) and `train_ppo_mjx.py` (more feature-rich MJX trainer with curriculum support and automatic validation).
- New features: curriculum of perturbation strengths (`PERTURB_CURRICULUM_RANGES`), recording of resume provenance (`resume_info.json` + copies in `save_root/resume_files`), and automatic validation runs when saving snapshots (best/stage).
- Reward defaults changed: notable sensor-related defaults were adjusted (see "Reward notes"). Always consult `reward.py` for exact defaults and environment variable overrides.

## System overview

- Observations: three tactile sensors (`sensor_p1_0`, `sensor_0_0`, `sensor_m1_0`) are stacked across `STACK_SIZE` frames. Optionally normalized by `RunningMeanStd` when `USE_OBS_NORMALIZATION=1`.
- Policy/value: shared ActorCritic MLPs (tanh activations). The actor outputs mean actions and keeps a learnable `log_std`; samples are squashed with `tanh` into [-1, 1].
- Action: a single continuous force along the world x-axis. The environment applies `action × ACTION_FORCE_MAGNITUDE`.
- Rollouts: on‑policy rollouts are collected, GAE is used for advantage estimates, and PPO performs minibatch updates.

## Recent changes (what to document)

1. Curriculum-based perturbations
	- `train_ppo_mjx.py` supports a perturbation curriculum via `PERTURB_CURRICULUM_RANGES` (list of `(min-max)` segments). The trainer samples startup perturbation magnitudes from the current stage range and can advance stages every `CURRICULUM_STAGE_EPISODES` episodes.
	- Example via environment variable:

	  PERTURB_CURRICULUM_RANGES="0-200;100-200;200-300;300-500"

	  Each `min-max` pair is a stage. The trainer parses this string and uses it instead of code defaults.

2. Resume provenance and snapshot metadata
	- When resuming from an existing run, `train_ppo_mjx.py` copies the resumed model/obs_rms into `save_root/resume_files/` and writes `resume_info.json` with timestamp, command args, and a snapshot of uppercase settings.

3. Automatic validation hooks
	- The trainer runs automatic validation when a new best model is saved and when a curriculum stage completes. Validation outputs are saved under `save_root/validation/` and include reward plots. This replaces the earlier ad-hoc validation block with a reusable `run_validation_pass` flow.

4. Reward parameter changes
	- Sensor-related defaults were adjusted in `reward.py` (example: `SENSOR_MATCH_BONUS` increased, tolerances changed). See `reward.py` for the exact values and environment variable overrides (you can override any constant using an environment variable with the same name).

## Reward notes

- Reward constants live in `reward.py`. Defaults can be overridden by environment variables. Notable parameters you might want to tune:
  - `ALIVE_BONUS` — granted while the agent is 'alive' (contact + tilt limits).
  - `FALL_PENALTY` — large negative reward on failure.
  - `SENSOR_MATCH_BONUS`, `SENSOR_MATCH_TOLERANCE`, `SENSOR_CONTACT_TOLERANCE` — govern the sensor-agreement bonus.

Always check `reward.py` for the current defaults; recent commits changed sensor defaults (e.g. `SENSOR_MATCH_BONUS` and `SENSOR_MATCH_TOLERANCE`).

## Curriculum / perturbation configuration

- Environment variable: `PERTURB_CURRICULUM_RANGES`. Format: semicolon-separated `min-max` ranges, e.g. `0-200;200-400;400-700`.
- If unset, `train_ppo_mjx.py` uses a built-in default list.
- `CURRICULUM_STAGE_EPISODES` controls how many completed episodes are required before the trainer considers advancing to the next stage (default present in `settings.py`).

## Resuming and provenance

- Resume: `python train_ppo_mjx.py --resume-dir pretrained/<run> --resume-tag best`.
- When resuming, the trainer copies the resume model and obs-rms into your new `save_root/resume_files/` and writes `resume_info.json` (timestamp, selected settings snapshot, cmd args). This helps reproduce experiments later.

## Validation and saved outputs

- Validation results (plots and numeric rewards) are written into `<save_root>/validation/` by the automatic validator.
- When a new best model or a stage snapshot is saved, the trainer optionally runs the validation flow and saves metrics and plots next to checkpoints.

## How to run

Train (classic):

```cmd
python train_box_ppo.py
```

Train (MJX trainer, with more features including curriculum and resume):

```cmd
python train_ppo_mjx.py [--resume-dir <path>] [--resume-tag best|last|snapshot]
```

Short smoke test:

```cmd
set SHORT_TEST=1
python train_ppo_mjx.py
```

Set a curriculum via environment variable (Windows cmd example):

```cmd
set PERTURB_CURRICULUM_RANGES=0-200;100-200;200-300;300-500
python train_ppo_mjx.py
```

Evaluate best checkpoint (example):

```cmd
python validate_best.py --model pretrained/latest/box_ppo_model_best.pt --obs_rms pretrained/latest/obs_rms_best.npz --episodes 5
```

## Key files

- `train_ppo_mjx.py` — main MJX trainer (curriculum, resume provenance, automatic validation).
- `train_box_ppo.py` — simpler single‑env trainer.
- `reward.py` — reward shaping constants (can be overridden with env vars).
- `settings.py` — shared defaults and derived dimensions; now includes curriculum parsing support.
- `validate_best.py`, `validate_compare.py`, `validate_random_perturb.py` — evaluation scripts.

## Where to look in the code

- Curriculum parsing and defaults: `settings.py` (look for `PERTURB_CURRICULUM_RANGES` and `CURRICULUM_STAGE_EPISODES`).
- Startup perturb sampling and curriculum advancement: `train_ppo_mjx.py` (`_sample_startup_perturbation`, `maybe_advance_curriculum`).
- Resume provenance and `resume_info.json`: `train_ppo_mjx.py` (resume handling section).
- Reward defaults and env overrides: `reward.py`.

## Notes & tips

- When experimenting with curriculum ranges, pick overlapping ranges to smoothly increase disturbance magnitude.
- Use `--resume-tag` to resume either the `best` or `last` snapshot; the trainer will copy resumed artifacts into your new run folder for traceability.

---

If you'd like, I can:

- Apply a small example `PERTURB_CURRICULUM_RANGES` to the README that mirrors your current config, or
- Add a short examples/ directory with a minimal run script demonstrating setting environment vars on Windows.

Changes applied: update README to add curriculum, resume and validation notes and correct reward/config references.
