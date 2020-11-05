from PhysicalModels import models, localization_error
import numpy as np


class Brownian(models.Models):

    # d_high = 0.6
    # d_low = 0.003

    d_high = 0.2
    d_low = 0.05

    def __init__(self, diffusion_coefficient):
        self.diffusion_coefficient = diffusion_coefficient

    @classmethod
    def create_random(cls):
        diffusion_coefficient = np.random.uniform(low=cls.d_low, high=cls.d_high)
        return cls(diffusion_coefficient=diffusion_coefficient)

    def simulate_track(self, track_length, track_time):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        for i in range(track_length):
            x[i] = x[i] * np.sqrt(2 * self.diffusion_coefficient * (track_time / track_length))
            y[i] = y[i] * np.sqrt(2 * self.diffusion_coefficient * (track_time / track_length))

        x = np.cumsum(x)
        y = np.cumsum(y)

        x, x_noisy, y, y_noisy = localization_error.add_noise_and_offset(track_length, x, y)

        t = np.linspace(0, track_time, track_length)

        return x_noisy, y_noisy, x, y, t

    def normalize_d_coefficient_to_net(self):
        delta_d = self.d_high - self.d_low
        return (1 / delta_d) * (self.diffusion_coefficient - self.d_low)

    @classmethod
    def denormalize_d_coefficient_to_net(cls, output_coefficient_net):
        delta_d = cls.d_high - cls.d_low
        return output_coefficient_net * delta_d + cls.d_low

    def get_d_coefficient(self):
        return self.diffusion_coefficient
