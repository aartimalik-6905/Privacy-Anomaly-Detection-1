import numpy as np

def detect_violence_spikes(
    speed,
    velocity_vectors,
    min_energy=0.02,          # 🔒 NEW: low-energy filter
    accel_multiplier=2.5,
    direction_threshold=0.7,
    min_events=4
):
    if len(speed) < 20:
        return False

    # --- ENERGY GATE (CRITICAL) ---
    motion_energy = np.mean(speed)
    if motion_energy < min_energy:
        return False  # walking / slow motion blocked here

    # --- Acceleration spikes ---
    accel = np.diff(speed, prepend=0)
    accel_std = np.std(accel)
    if accel_std == 0:
        return False

    accel_spikes = accel > (accel_multiplier * accel_std)

    # --- Direction change spikes ---
    norms = np.linalg.norm(velocity_vectors, axis=1, keepdims=True) + 1e-6
    directions = velocity_vectors / norms
    direction_change = np.linalg.norm(np.diff(directions, axis=0), axis=1)
    direction_spikes = direction_change > direction_threshold

    violent_events = np.sum(accel_spikes) + np.sum(direction_spikes)

    return violent_events >= min_events
