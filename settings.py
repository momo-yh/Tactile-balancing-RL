import math
import os

# Shared environment-configurable parameters used by training and validation

# Observation / stacking
STACK_SIZE = int(os.environ.get("STACK_SIZE", "32"))
OBS_DIM = int(os.environ.get("OBS_DIM", "2"))
STATE_DIM = OBS_DIM * STACK_SIZE

# Network / action dimensions
ACTION_DIM = int(os.environ.get("ACTION_DIM", "1"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "64"))

# Force magnitudes
ACTION_FORCE_MAGNITUDE = float(os.environ.get("ACTION_FORCE_MAGNITUDE", "50.0"))

# Startup perturbation settings
PERTURB_FORCE_SCALE = float(os.environ.get("PERTURB_FORCE_SCALE", "100.0"))
PERTURB_APPLY_STEPS = int(os.environ.get("PERTURB_APPLY_STEPS", "1"))
PERTURB_START_DELAY = int(os.environ.get("PERTURB_START_DELAY", "200"))
PERTURB_RESAMPLE_EPISODES = int(os.environ.get("PERTURB_RESAMPLE_EPISODES", "1"))

ACTION_START_DELAY = int(os.environ.get("ACTION_START_DELAY", "150"))

# Tilt-based termination threshold (degrees and radians)
TILT_LIMIT_DEGREES = float(os.environ.get("TILT_LIMIT_DEGREES", "15.0"))
TILT_LIMIT_RADIANS = math.radians(TILT_LIMIT_DEGREES)

# World / episode limits
BOUNDARY_LIMIT = float(os.environ.get("BOUNDARY_LIMIT", "2.0"))

# Observation normalization toggle (0 disables RunningMeanStd updates/usage)
USE_OBS_NORMALIZATION = os.environ.get("USE_OBS_NORMALIZATION", "0") == "1"

# Perturbation curriculum configuration (course learning)
# Format: list of (min_force, max_force) tuples applied sequentially across stages.
# You can override via environment variable PERTURB_CURRICULUM_RANGES using
# semicolon-separated "min-max" entries, e.g. "0-50;50-100;100-200;200-500".
_default_curriculum = [(0.0, 200.0)]
_env_curriculum = os.environ.get("PERTURB_CURRICULUM_RANGES")
if _env_curriculum:
    parsed = []
    for segment in _env_curriculum.split(';'):
        segment = segment.strip()
        if not segment:
            continue
        try:
            min_s, max_s = segment.split('-')
            parsed.append((float(min_s), float(max_s)))
        except ValueError:
            raise ValueError(f"Invalid PERTURB_CURRICULUM_RANGES segment '{segment}'. Expected 'min-max'.")
    if parsed:
        PERTURB_CURRICULUM_RANGES = parsed
    else:
        PERTURB_CURRICULUM_RANGES = _default_curriculum
else:
    PERTURB_CURRICULUM_RANGES = _default_curriculum

# Number of completed episodes before advancing to the next curriculum stage.
# Can be overridden with CURRICULUM_STAGE_EPISODES environment variable.
CURRICULUM_STAGE_EPISODES = int(os.environ.get("CURRICULUM_STAGE_EPISODES", "1000"))

