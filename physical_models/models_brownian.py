from physical_models import models, models_noise
import numpy as np


class Brownian(models.Models):

    def __init__(self, diffusion_coefficient):
        self.diffusion_coefficient = diffusion_coefficient

    @classmethod
    def create_random(cls):
        diffusion_coefficient = np.random.uniform(low=0.001, high=0.6)
        return cls(diffusion_coefficient=diffusion_coefficient)

    def simulate_track(self, track_length, track_time):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        for i in range(track_length):
            x[i] = x[i] * np.sqrt(2 * self.diffusion_coefficient * (track_time/track_length))
            y[i] = y[i] * np.sqrt(2 * self.diffusion_coefficient * (track_time / track_length))

        x = np.cumsum(x)
        y = np.cumsum(y)

        x, x_noisy, y, y_noisy = models_noise.add_noise_and_offset(track_length, x, y)

        t = np.linspace(0, track_time, track_length)

        return x_noisy, y_noisy, x, y, t
