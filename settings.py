import math
import os

# Shared environment-configurable parameters used by training and validation

# Observation / stacking
STACK_SIZE = int(os.environ.get("STACK_SIZE", "8"))
OBS_DIM = int(os.environ.get("OBS_DIM", "3"))
STATE_DIM = OBS_DIM * STACK_SIZE

# Network / action dimensions
ACTION_DIM = int(os.environ.get("ACTION_DIM", "1"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "64"))

# Force magnitudes
ACTION_FORCE_MAGNITUDE = float(os.environ.get("ACTION_FORCE_MAGNITUDE", "50.0"))

# Startup perturbation settings
PERTURB_FORCE_SCALE = float(os.environ.get("PERTURB_FORCE_SCALE", "100.0"))
PERTURB_APPLY_STEPS = int(os.environ.get("PERTURB_APPLY_STEPS", "1"))
PERTURB_START_DELAY = int(os.environ.get("PERTURB_START_DELAY", "150"))
PERTURB_RESAMPLE_EPISODES = int(os.environ.get("PERTURB_RESAMPLE_EPISODES", "1"))

ACTION_START_DELAY = int(os.environ.get("ACTION_START_DELAY", str(PERTURB_START_DELAY)))

# Tilt-based termination threshold (degrees and radians)
TILT_LIMIT_DEGREES = float(os.environ.get("TILT_LIMIT_DEGREES", "15.0"))
TILT_LIMIT_RADIANS = math.radians(TILT_LIMIT_DEGREES)

# World / episode limits
BOUNDARY_LIMIT = float(os.environ.get("BOUNDARY_LIMIT", "2.0"))

# Observation normalization toggle (0 disables RunningMeanStd updates/usage)
USE_OBS_NORMALIZATION = os.environ.get("USE_OBS_NORMALIZATION", "0") == "1"

