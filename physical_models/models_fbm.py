import numpy as np
from scipy import fftpack
from . import models
from . import models_noise


class FBM(models.Models):
    sub_diff_min_max = [0.1, 0.42]
    super_diff_min_max = [0.58, 0.9]

    def __init__(self, hurst_exp):
        self.hurst_exp = hurst_exp

    @classmethod
    def create_random(cls):
        fbm_type = np.random.choice(["subdiffusive", "superdiffusive", "brownian"])
        if fbm_type == "subdiffusive":
            model = cls.create_random_subdiffusive()
        elif fbm_type == "superdiffusive":
            model = cls.create_random_superdiffusive()
        else:
            model = cls.create_random_brownian()
        return model

    @classmethod
    def create_random_superdiffusive(cls, hurst_exp=None):
        if hurst_exp is not None:
            assert (cls.super_diff_min_max[0] <= hurst_exp <= cls.super_diff_min_max[
                1]), "Invalid Hurst Exponent"
            model = cls(hurst_exp=hurst_exp)

        else:
            random_hurst_exp = np.random.uniform(low=cls.super_diff_min_max[0], high=cls.super_diff_min_max[1])
            model = cls(hurst_exp=random_hurst_exp)
        return model

    @classmethod
    def create_random_subdiffusive(cls, hurst_exp=None):
        if hurst_exp is not None:
            assert (cls.sub_diff_min_max[0] <= hurst_exp <= cls.sub_diff_min_max[
                1]), "Invalid Hurst Exponent"
            model = cls(hurst_exp=hurst_exp)
        else:
            random_hurst_exp = np.random.uniform(low=cls.sub_diff_min_max[0], high=cls.sub_diff_min_max[1])
            model = cls(hurst_exp=random_hurst_exp)
        return model

    @classmethod
    def create_random_brownian(cls, use_exact_exp=False):
        if use_exact_exp:
            model = cls(hurst_exp=0.5)
        else:
            random_brownian_hurst_exp = np.random.uniform(low=cls.sub_diff_min_max[1], high=cls.super_diff_min_max[0])
            model = cls(hurst_exp=random_brownian_hurst_exp)
        return model

    def simulate_track(self, track_length, track_time):

        r = np.zeros(track_length + 1)  # first row of circulant matrix
        r[0] = 1
        idx = np.arange(1, track_length + 1, 1)
        r[idx] = 0.5 * ((idx + 1) ** (2 * self.hurst_exp) - 2 * idx ** (2 * self.hurst_exp) + (idx - 1) ** (
                    2 * self.hurst_exp))
        r = np.concatenate((r, r[np.arange(len(r) - 2, 0, -1)]))

        # get eigenvalues through fourier transform
        lamda = np.real(fftpack.fft(r)) / (2 * track_length)

        # get trajectory using fft: dimensions assumed uncoupled
        x = fftpack.fft(np.sqrt(lamda) * (
                    np.random.normal(size=(2 * track_length)) + 1j * np.random.normal(size=(2 * track_length))))
        x = track_length ** (-self.hurst_exp) * np.cumsum(np.real(x[:track_length]))  # rescale
        x = ((track_time ** self.hurst_exp) * x)  # resulting trajectory in x

        y = fftpack.fft(np.sqrt(lamda) * (
                    np.random.normal(size=(2 * track_length)) + 1j * np.random.normal(size=(2 * track_length))))
        y = track_length ** (-self.hurst_exp) * np.cumsum(np.real(y[:track_length]))  # rescale
        y = ((track_time ** self.hurst_exp) * y)  # resulting trajectory in y

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

        t = np.arange(0, track_length, 1) / track_length
        t = t * track_time  # scale for final time T

        return x_noisy, y_noisy, x, y, t

    def get_diffusion_type(self):
        if self.sub_diff_min_max[0] <= self.hurst_exp <= self.sub_diff_min_max[1]:
            return "subdiffusive"
        elif self.sub_diff_min_max[1] < self.hurst_exp < self.super_diff_min_max[0]:
            return "brownian"
        else:
            return "superdiffusive"
