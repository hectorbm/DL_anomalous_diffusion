import numpy as np
from scipy import fftpack
from . import models
from . import localization_error

FILE_pixel_size = 106  # nm


class FBM(models.Models):
    sub_diff_min_max = [0.1, 0.42]
    super_diff_min_max = [0.58, 0.9]

    def __init__(self, hurst_exp):
        self.hurst_exp = hurst_exp

    @classmethod
    def create_random(cls):
        fbm_type = np.random.choice(["Subdiffusive", "Superdiffusive", "Brownian"])
        if fbm_type == "Subdiffusive":
            model = cls.create_random_subdiffusive()
        elif fbm_type == "Superdiffusive":
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
        x = x * FILE_pixel_size
        y = y * FILE_pixel_size

        x, x_noisy, y, y_noisy = localization_error.add_noise_and_offset(track_length, x, y)

        t = np.linspace(0, track_time, track_length)

        return x_noisy, y_noisy, x, y, t

    def get_diffusion_type(self):
        if self.sub_diff_min_max[0] <= self.hurst_exp <= self.sub_diff_min_max[1]:
            return "Subdiffusive"
        elif self.sub_diff_min_max[1] < self.hurst_exp < self.super_diff_min_max[0]:
            return "Brownian"
        else:
            return "Superdiffusive"
