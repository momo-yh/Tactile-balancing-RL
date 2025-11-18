import numpy as np

SENSOR_NAMES = [
    'sensor_p1_p1', 'sensor_0_p1', 'sensor_m1_p1',
    'sensor_p1_0', 'sensor_0_0', 'sensor_m1_0',
    'sensor_p1_m1', 'sensor_0_m1', 'sensor_m1_m1',
]

MID_ROW_INDICES = (3, 4, 5)


def _box_body(data):
    return data.body('box_body')


def tactile_sensors(data):
    """Return array of z-axis forces from each tactile site (length 9)."""
    readings = []
    for name in SENSOR_NAMES:
        try:
            sensor = np.asarray(data.sensor(name).data, dtype=np.float32).ravel()
        except Exception:
            sensor = np.zeros(3, dtype=np.float32)
        if sensor.size >= 3:
            readings.append(sensor[2])
        elif sensor.size > 0:
            readings.append(sensor[-1])
        else:
            readings.append(0.0)
    return np.asarray(readings, dtype=np.float32)


def mid_row_sensors(data):
    """Return the three middle-row tactile readings (positive x, center, negative x)."""
    sensors = tactile_sensors(data)
    try:
        return sensors[list(MID_ROW_INDICES)].astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def compute_tilt_angle(data):
    """Signed tilt (radians) around the world y-axis."""
    try:
        quat = _box_body(data).xquat
        z_axis = np.array([
            2 * (quat[1] * quat[3] - quat[0] * quat[2]),
            2 * (quat[2] * quat[3] + quat[0] * quat[1]),
            1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
        ], dtype=np.float32)
        return float(np.arctan2(z_axis[0], z_axis[2]))
    except Exception:
        return 0.0


def compute_tilt_rate(data):
    """Angular velocity (radians/s) around the y-axis."""
    try:
        spatial_vel = _box_body(data).cvel
        if spatial_vel.size >= 2:
            return float(spatial_vel[1])
    except Exception:
        pass
    return 0.0


def build_observation(data):
    """Observation vector with tilt proxy and raw angular velocity."""
    middle_row = mid_row_sensors(data)
    tilt_proxy = float(middle_row[0] - middle_row[2])
    angular_vel = compute_tilt_rate(data)
    return np.asarray([tilt_proxy, float(angular_vel)], dtype=np.float32)
