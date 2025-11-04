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

- `STACK_SIZE`: 8 by default (frames to stack).
- `OBS_DIM`: 3 by default (three middle‑row tactile sensors).
- `STATE_DIM = OBS_DIM × STACK_SIZE`: 24 by default (3 × 8).
- `ACTION_DIM`: 1 (apply force along world x). The network outputs in [-1, 1], then the force applied in MuJoCo is `action × ACTION_FORCE_MAGNITUDE` (default 50.0).

The `StateStacker` appends the latest observation each step so the network receives a flattened state of shape `(STATE_DIM,)`.

## 5) Reward shaping (high‑level)

- Alive bonus (`ALIVE_BONUS`) when contact and tilt are within limits.
- Angular velocity penalty (`ANGULAR_VEL_PENALTY` × tilt_rate²) to suppress oscillations.
- Action energy penalty (`ACTION_ENERGY_PENALTY` × force²) to discourage excessive forces.
- Sensor agreement bonus (`SENSOR_MATCH_BONUS`) when the three tactile sensors agree, governed by `SENSOR_MATCH_TOLERANCE`.
- Failure penalty (`FALL_PENALTY`) when the tilt or position leaves the safe region.

See `reward.py` for exact constants and defaults.

## 6) Training hyperparameters (defaults in the script)

- `MAX_TRAINING_STEPS`: 5_000_000 total steps
- `ROLLOUT_LENGTH`: 512 steps per PPO update (256 when `SHORT_TEST=1`)
- `BATCH_SIZE`: 128 (minibatch size)
- `PPO_EPOCHS`: 16 (epochs per rollout)
- `LR`: 1e‑4 (initial learning rate)
- `ENTROPY_COEF`: 0.01
- `MAX_EPISODE_STEPS`: 2000 (time limit per episode)

PPO update sample count per update (single env): `ROLLOUT_LENGTH`. With defaults: 512 samples → 512/128 = 4 minibatches per update.

The training loop also uses delayed action start (`ACTION_START_DELAY`) and startup perturbations (`PERTURB_*`) to test robustness.

## 7) Environment variables

- `SHORT_TEST=1`: quick smoke test (shrinks `MAX_TRAINING_STEPS` and `ROLLOUT_LENGTH`).
- `USE_OBS_NORMALIZATION=1`: enable online observation normalization via `RunningMeanStd`.
- `LR_DECAY_FRACTION`: when to trigger LR decay (default 0.75 of expected updates).
- `LR_DECAY_GAMMA`: multiplicative LR drop (default 0.1; use 1.0 to disable).
- `PERIODIC_SAVE_STEPS`: step interval for periodic “last” checkpoints (default set in code).

## 8) Learning‑rate scheduling

`train_box_ppo.py` enables a late‑stage LR decay via `torch.optim.lr_scheduler.MultiStepLR`.

- The script estimates the total number of PPO updates (`MAX_TRAINING_STEPS // ROLLOUT_LENGTH`), places one milestone at `LR_DECAY_FRACTION`, then applies `LR_DECAY_GAMMA`.
- Example: `1e‑4 → 1e‑5` at ~75% of the run with defaults.

## 9) How to run

Train:

```bash
python train_box_ppo.py
```

Short smoke test:

```bash
SHORT_TEST=1 python train_box_ppo.py
```

Resume from a previous run:

```bash
python train_box_ppo.py --resume-dir pretrained/<your-run> --resume-tag best
```

Evaluate the best checkpoint (deterministic replay):

```bash
python validate_best.py \
	--model pretrained/latest/box_ppo_model_best.pt \
	--obs_rms pretrained/latest/obs_rms_best.npz \
	--episodes 5
```

Compare controller vs open‑loop:

```bash
python validate_compare.py --model pretrained/latest/box_ppo_model_best.pt --episodes 3
```

Random continuous perturbation validation (robustness test):

```bash
python validate_random_perturb.py \
	--model pretrained/latest/box_ppo_model_best.pt \
	--obs_rms pretrained/latest/obs_rms_best.npz \
	--episodes 1 \
	--max_steps 1500 \
	--noise_type gaussian \
	--noise_scale 5.0
```

Notes:
- `--noise_type` can be `gaussian`, `uniform`, or `ou` (Ornstein–Uhlenbeck). Use `--ou_theta/--ou_mu` to tune OU.
- The random force is added to the agent’s x‑force each step (continuous disturbance), then applied via `data.xfrc_applied[box_body_id][0]`.

Notes:
- `validate_best.py` supports `--max_steps`, `--render`, and `--force_given` (to fix the startup perturbation force), which can help produce controlled, repeatable plots.
- At the end of training, the script launches a passive MuJoCo viewer for a quick smoke evaluation. In headless environments, the viewer import will fail gracefully and evaluation will continue without rendering.

## 10) What’s inside a minibatch

During PPO updates, data come from a single‑env rollout of length `ROLLOUT_LENGTH`, shuffled and split into minibatches:

- `obs_b`: `(batch_size, STATE_DIM)` float32; observations are sanitized (`nan/inf → finite`) and optionally normalized.
- `actions_b`: `(batch_size, ACTION_DIM)` float32 in [-1, 1] (network outputs); the environment applies `× ACTION_FORCE_MAGNITUDE`.
- `b_old_log_probs`, `b_advantages`, `b_returns`: `(batch_size,)` float32.

With defaults: 512 samples per update, batch size 128 → 4 minibatches.

To peek at real data, you can either:

1) Temporarily print in the training loop near `stored = agent.store_transition(...)` (lightweight).
2) Write a tiny script to create a `PPOAgent`, push a few fake transitions into `RolloutBuffer`, then call `get_minibatches(batch_size)` and print the first batch.

I can provide either on request.

## 11) Debugging tips

- If learning stalls, revisit `LR` and `ROLLOUT_LENGTH` first. Batch size and epochs also change the optimization dynamics.
- If non‑finite values appear (NaN/inf), the script will warn and skip polluted minibatches. Check whether the culprit is `obs`, `actions`, or network outputs.
- PPO update cadence: `agent.store_transition(...)` returns `True` every `ROLLOUT_LENGTH` steps, which triggers `agent.update(...)`.
- When using observation normalization, make sure you load the matching `obs_rms_*.npz` when resuming or evaluating to keep distributions consistent.

## 12) Repro and artifacts

- Artifacts are saved under `pretrained/<timestamp>_TILTXX_RL/`, and mirrored to `pretrained/latest/` for convenience.
- A simple reward curve (`box_ppo_rewards.png`) is written at the end of training.
- To compare runs, keep `ROLLOUT_LENGTH`, `LR`, `USE_OBS_NORMALIZATION`, and perturbation settings the same.

---

Need a ready‑to‑run inspection script or temporary prints added to `train_box_ppo.py` to show a real minibatch? Tell me which you prefer and I’ll add it.
