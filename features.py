import numpy as np

SENSOR_NAMES = [
    'sensor_p1_p1', 'sensor_0_p1', 'sensor_m1_p1',
    'sensor_p1_0', 'sensor_0_0', 'sensor_m1_0',
    'sensor_p1_m1', 'sensor_0_m1', 'sensor_m1_m1',
]


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
        spatial_vel = _box_body(data).xvel
        if spatial_vel.size >= 5:
            return float(spatial_vel[4])
    except Exception:
        pass
    return 0.0


def compute_position_velocity(data):
    """Return (x_position, x_velocity) in world coordinates."""
    try:
        x_pos = float(_box_body(data).xpos[0])
    except Exception:
        x_pos = 0.0
    try:
        spatial_vel = _box_body(data).xvel
        x_vel = float(spatial_vel[0]) if spatial_vel.size >= 1 else 0.0
    except Exception:
        x_vel = 0.0
    return x_pos, x_vel


def build_observation(data):
    """Observation vector consisting only of the middle-row tactile sensors."""
    sensors = tactile_sensors(data)
    middle_row = sensors[3:6].astype(np.float32)
    return middle_row
