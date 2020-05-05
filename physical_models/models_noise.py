import numpy as np


def add_noise(track_length):
    # New error formula
    mean_error = 40
    sigma_error = 10
    error_x = np.random.normal(loc=mean_error / 2, scale=sigma_error / 2, size=track_length)
    error_x_sign = np.random.choice([-1, 1], size=track_length)
    error_y = np.random.normal(loc=mean_error / 2, scale=sigma_error / 2, size=track_length)
    error_y_sign = np.random.choice([-1, 1], size=track_length)
    return error_x * error_x_sign, error_y * error_y_sign


def add_noise_and_offset(track_length, x, y):
    noise_x, noise_y = add_noise(track_length)
    x_noisy = x + noise_x
    y_noisy = y + noise_y
    if np.min(x_noisy) < np.min(x) and np.min(x_noisy) < 0:
        min_noisy_x = np.absolute(np.min(x_noisy))
        x_noisy = x_noisy + min_noisy_x  # Convert to positive
        x = x + min_noisy_x
    if np.min(x_noisy) > np.min(x) and np.min(x) < 0:
        min_x = np.absolute(np.min(x))
        x_noisy = x_noisy + min_x  # Convert to positive
        x = x + min_x
    if np.min(y_noisy) < np.min(y) and np.min(y_noisy) < 0:
        min_noisy_y = np.absolute(np.min(y_noisy))
        y_noisy = y_noisy + min_noisy_y  # Convert to positive
        y = y + min_noisy_y
    if np.min(y_noisy) > np.min(y) and np.min(y) < 0:
        min_y = np.absolute(np.min(y))
        y_noisy = y_noisy + min_y  # Convert to positive
        y = y + min_y
    offset_x = np.ones(shape=track_length) * np.random.uniform(low=0, high=(
            10000 - np.minimum(np.max(x), np.max(x_noisy))))
    offset_y = np.ones(shape=track_length) * np.random.uniform(low=0, high=(
            10000 - np.minimum(np.max(y), np.max(y_noisy))))
    x = x + offset_x
    y = y + offset_y
    x_noisy = x_noisy + offset_x
    y_noisy = y_noisy + offset_y
    return x, x_noisy, y, y_noisy
