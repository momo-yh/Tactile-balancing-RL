import os
import numpy as np

from features import compute_tilt_angle, compute_tilt_rate

# Default reward coefficients (can be overridden via environment variables)
ALIVE_BONUS = float(os.environ.get("ALIVE_BONUS", "1.0"))
FALL_PENALTY = float(os.environ.get("FALL_PENALTY", "10.0"))
ANGVEL_BONUS = float(
    os.environ.get(
        "ANGVEL_BONUS",
        os.environ.get("SENSOR_MATCH_BONUS", "1.0"),
    )
)
ANGVEL_TOLERANCE = float(
    os.environ.get(
        "ANGVEL_TOLERANCE",
        os.environ.get("SENSOR_MATCH_TOLERANCE", "0.1"),
    )
)


def compute_reward(data, action_force, tilt_limit_rad, out_of_bounds=False):
    """Compute PPO reward for the balancing task.

        Components:
            + alive bonus while |tilt| <= tilt_limit_rad and within bounds
            + angular velocity bonus that shrinks linearly with |d(tilt)/dt|
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
        reward = ALIVE_BONUS
        if ANGVEL_BONUS > 0.0:
            angular_velocity = abs(float(compute_tilt_rate(data)))
            normalized_rate = angular_velocity / max(ANGVEL_TOLERANCE, 1e-8)
            linear_factor = float(np.clip(1.0 - normalized_rate, 0.0, 1.0))
            reward += ANGVEL_BONUS * linear_factor
        return float(reward), True

    return -FALL_PENALTY, False


