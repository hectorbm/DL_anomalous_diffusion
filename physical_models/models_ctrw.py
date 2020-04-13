import numpy as np
from . import models
from . import models_noise


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class CTRW(models.Models):
    min_alpha = 0.1
    max_alpha = 0.9

    def __init__(self, alpha):
        assert (self.min_alpha <= alpha <= self.max_alpha), "Invalid alpha parameter"
        self.alpha = alpha
        self.beta = 0.5
        self.gamma = 1

    @classmethod
    def create_random(cls):
        random_alpha = np.random.uniform(low=cls.min_alpha, high=cls.max_alpha)
        model = cls(alpha=random_alpha)
        return model

    def mittag_leffler_rand(self, track_length):
        # Generate mittag-leffler random numbers
        t = -np.log(np.random.uniform(size=[track_length, 1]))
        u = np.random.uniform(size=[track_length, 1])
        w = np.sin(self.beta * np.pi) / np.tan(self.beta * np.pi * u) - np.cos(self.beta * np.pi)
        t = t * (w ** 1 / self.beta)
        t = self.gamma * t
        return t

    def symmetric_alpha_levy(self, track_length):
        alpha_levy_dist = 2
        gamma_levy_dist = self.gamma ** (self.alpha / 2)
        # Generate symmetric alpha-levi random numbers
        u = np.random.uniform(size=[track_length, 1])
        v = np.random.uniform(size=[track_length, 1])

        phi = np.pi * (v - 0.5)
        w = np.sin(alpha_levy_dist * phi) / np.cos(phi)
        z = -1 * np.log(u) * np.cos(phi)
        z = z / np.cos((1 - alpha_levy_dist) * phi)
        x = gamma_levy_dist * w * z ** (1 - (1 / alpha_levy_dist))

        return x

    def simulate_track(self, track_length, track_time):

        jumps_x = self.mittag_leffler_rand(track_length)
        raw_time_x = np.cumsum(jumps_x)
        t_x = raw_time_x * track_time / np.max(raw_time_x)
        t_x = np.reshape(t_x, [len(t_x), 1])

        jumps_y = self.mittag_leffler_rand(track_length)
        raw_time_y = np.cumsum(jumps_y)
        t_y = raw_time_y * track_time / np.max(raw_time_y)
        t_y = np.reshape(t_y, [len(t_y), 1])

        x = self.symmetric_alpha_levy(track_length)
        x = np.cumsum(x)
        x = np.reshape(x, [len(x), 1])

        y = self.symmetric_alpha_levy(track_length)
        y = np.cumsum(y)
        y = np.reshape(y, [len(y), 1])

        t_out = np.arange(0, track_length, 1) * track_time / track_length
        x_out = np.zeros([track_length, 1])
        y_out = np.zeros([track_length, 1])
        for i in range(track_length):
            x_out[i, 0] = x[find_nearest(t_x, t_out[i]), 0]
            y_out[i, 0] = y[find_nearest(t_y, t_out[i]), 0]

        x = x_out[:, 0]
        y = y_out[:, 0]
        t = t_out

        # Scale to 10.000 nm * 10.000 nm
        if np.min(x) < 0:
            x = x + np.absolute(np.min(x))  # Add offset to x
        if np.min(y) < 0:
            y = y + np.absolute(np.min(y))  # Add offset to y
        # Scale to nm and add a random offset
        if np.max(x) != 0:
            x = x * (1 / np.max(x)) * np.min([10000, ((track_length ** 1.1) * np.random.uniform(low=3, high=4))])
        else:
            x = x * np.min([10000, ((track_length ** 1.1) * np.random.uniform(low=3, high=4))])
        if np.max(y) != 0:
            y = y * (1 / np.max(y)) * np.min([10000, ((track_length ** 1.1) * np.random.uniform(low=3, high=4))])
        else:
            y = y * np.min([10000, ((track_length ** 1.1) * np.random.uniform(low=3, high=4))])

        noise_x, noise_y = models_noise.add_noise(track_length)

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

        return x_noisy, y_noisy, x, y, t
