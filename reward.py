import os
import numpy as np

from features import compute_tilt_angle, mid_row_sensors

# Default reward coefficients (can be overridden via environment variables)
ALIVE_BONUS = float(os.environ.get("ALIVE_BONUS", "1.0"))
FALL_PENALTY = float(os.environ.get("FALL_PENALTY", "10.0"))
SENSOR_MATCH_BONUS = float(os.environ.get("SENSOR_MATCH_BONUS", "1"))
SENSOR_MATCH_TOLERANCE = float(os.environ.get("SENSOR_MATCH_TOLERANCE", "1"))
SENSOR_CONTACT_TOLERANCE = float(os.environ.get("SENSOR_CONTACT_TOLERANCE", "0.01"))


def compute_reward(data, action_force, tilt_limit_rad, out_of_bounds=False):
    """Compute PPO reward for the balancing task.

    Components:
      + alive bonus while |tilt| <= tilt_limit_rad and within bounds
      - fall penalty once the agent exceeds the tilt limit or leaves bounds

    Args:
        data: MuJoCo data handle for the current simulation step.
        action_force: scalar force applied along x (Newtons).
        tilt_limit_rad: maximum allowed tilt magnitude before failure.
        out_of_bounds: optional flag indicating world boundary violation.

    Returns:
        Tuple (reward, alive) where alive=False indicates episode termination.
    """
    tilt_angle = compute_tilt_angle(data)

    alive = (abs(tilt_angle) <= tilt_limit_rad) and (not out_of_bounds)

    if alive:
        sensors = mid_row_sensors(data).astype(np.float32)
        sensors = np.nan_to_num(sensors, nan=0.0, posinf=0.0, neginf=0.0)

        reward = ALIVE_BONUS
        if SENSOR_MATCH_BONUS > 0.0:
            spread = float(np.max(sensors) - np.min(sensors))
            match_score = max(0.0, 1.0 - spread / max(SENSOR_MATCH_TOLERANCE, 1e-8))
            reward += SENSOR_MATCH_BONUS * match_score
        return float(reward), True

    return -FALL_PENALTY, False


