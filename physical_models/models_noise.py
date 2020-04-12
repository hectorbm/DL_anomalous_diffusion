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
