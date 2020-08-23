import numpy as np
from physical_models import models
from physical_models import models_noise


def denormalize_d_coefficient_to_net(output_coefficient_net, state_number):
    assert (state_number == 0 or state_number == 1), "Not a valid state"
    delta_d0 = TwoStateDiffusion.d0_high - TwoStateDiffusion.d0_low
    delta_d1 = TwoStateDiffusion.d1_high - TwoStateDiffusion.d1_low
    if state_number == 0:
        return output_coefficient_net * delta_d0 + TwoStateDiffusion.d0_low
    else:
        return output_coefficient_net * delta_d1 + TwoStateDiffusion.d1_low


class TwoStateDiffusion(models.Models):
    """
    State-0: Free Diffusion
    State-1: Confined Diffusion
    """

    # http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-sim-BM.pdf

    d0_low = 0.05
    d0_high = 0.2
    d1_low = 0.001
    d1_high = 0.05

    def __init__(self, k_state0, k_state1, d_state0, d_state1):
        self.k_state0 = k_state0
        self.k_state1 = k_state1
        self.D_state0 = d_state0 * 1000000  # Convert from um^2 -> nm^2
        self.D_state1 = d_state1 * 1000000

    @classmethod
    def create_random(cls):
        # k_state(i) dimensions = 1 / frame
        # D_state(i) dimensions = um^2 * s^(-beta)

        d_state0 = np.random.uniform(low=cls.d0_low, high=cls.d0_high)
        d_state1 = np.random.uniform(low=cls.d1_low, high=cls.d1_high)
        k_state0 = np.random.uniform(low=0.01, high=0.08)
        k_state1 = np.random.uniform(low=0.007, high=0.2)
        model = cls(k_state0, k_state1, d_state0, d_state1)
        return model

    @classmethod
    def create_with_coefficients(cls, k_state0, k_state1, d_state0, d_state1):
        assert (0.05 <= d_state0 <= 0.3), "Invalid Diffusion coefficient state-0"
        assert (0.001 <= d_state1 <= 0.05), "Invalid Diffusion coefficient state-1"
        assert (0.01 <= k_state0 <= 0.08), "Invalid switching rate state-0"
        assert (0.007 <= k_state0 <= 0.2), "Invalid switching rate state-1"
        return cls(k_state0, k_state1, d_state0, d_state1)

    def get_d_state0(self):
        return self.D_state0 / 1000000

    def get_d_state1(self):
        return self.D_state1 / 1000000

    def normalize_d_coefficient_to_net(self, state_number):
        assert (state_number == 0 or state_number == 1), "Not a valid state"
        delta_d0 = self.d0_high - self.d0_low
        delta_d1 = self.d1_high - self.d1_low
        if state_number == 0:
            return (1 / delta_d0) * (self.get_d_state0() - self.d0_low)
        else:
            return (1 / delta_d1) * (self.get_d_state1() - self.d1_low)

    def denormalize_d_coefficient_to_net(self, state_number):
        assert (state_number == 0 or state_number == 1), "Not a valid state"
        delta_d0 = self.d0_high - self.d0_low
        delta_d1 = self.d1_high - self.d1_low
        if state_number == 0:
            return self.normalize_d_coefficient_to_net(state_number=0) * delta_d0 + self.d0_low
        else:
            return self.normalize_d_coefficient_to_net(state_number=1) * delta_d1 + self.d1_low

    def simulate_track(self, track_length, track_time):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        state, switching = self.simulate_states(track_length)

        for i in range(track_length):
            if state[i] == 0:
                x[i] = x[i] * np.sqrt(2 * self.D_state0 * (track_time / track_length))
                y[i] = y[i] * np.sqrt(2 * self.D_state0 * (track_time / track_length))
            else:
                x[i] = x[i] * np.sqrt(2 * self.D_state1 * (track_time / track_length))
                y[i] = y[i] * np.sqrt(2 * self.D_state1 * (track_time / track_length))

        x = np.cumsum(x)
        y = np.cumsum(y)

        x, x_noisy, y, y_noisy = models_noise.add_noise_and_offset(track_length, x, y)

        t = self.simulate_track_time(track_length, track_time)

        return x_noisy, y_noisy, x, y, t, state, switching

    def simulate_states(self, track_length):
        # Residence time
        res_time0 = 1 / self.k_state0
        res_time1 = 1 / self.k_state1
        # Compute each t_state according to exponential laws
        t_state0 = np.random.exponential(scale=res_time0, size=track_length)
        t_state1 = np.random.exponential(scale=res_time1, size=track_length)
        # Set initial t_state for each state
        t_state0_next = 0
        t_state1_next = 0
        # Pick an initial state from a random choice
        current_state = np.random.choice([0, 1])
        # Detect real switching behavior
        switching = ((current_state == 0) and (int(np.ceil(t_state0[t_state0_next])) < track_length)) or (
                (current_state == 1) and (int(np.ceil(t_state1[t_state1_next])) < track_length))
        # Fill state array
        state = np.zeros(shape=track_length)
        i = 0
        while i < track_length:
            if current_state == 1:
                current_state_length = int(np.ceil(t_state1[t_state1_next]))

                if (current_state_length + i) < track_length:
                    state[i:(i + current_state_length)] = np.ones(shape=current_state_length)
                else:
                    state[i:track_length] = np.ones(shape=(track_length - i))

                current_state = 0  # Set state from 1->0
            else:
                current_state_length = int(np.ceil(t_state0[t_state0_next]))
                current_state = 1  # Set state from 0->1

            i += current_state_length
        return state, switching

    def simulate_track_only_state0(self, track_length, track_time):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        for i in range(track_length):
            x[i] = x[i] * np.sqrt(2 * self.D_state0 * (track_time / track_length))
            y[i] = y[i] * np.sqrt(2 * self.D_state0 * (track_time / track_length))

        x = np.cumsum(x)
        y = np.cumsum(y)

        x, x_noisy, y, y_noisy = models_noise.add_noise_and_offset(track_length, x, y)

        t = self.simulate_track_time(track_length, track_time)

        return x_noisy, y_noisy, x, y, t

    def simulate_track_only_state1(self, track_length, track_time):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        for i in range(track_length):
            x[i] = x[i] * np.sqrt(2 * self.D_state1 * (track_time / track_length))
            y[i] = y[i] * np.sqrt(2 * self.D_state1 * (track_time / track_length))

        x = np.cumsum(x)
        y = np.cumsum(y)

        x, x_noisy, y, y_noisy = models_noise.add_noise_and_offset(track_length, x, y)

        t = self.simulate_track_time(track_length, track_time)

        return x_noisy, y_noisy, x, y, t

    def simulate_track_time(self, track_length, track_time):
        t = np.linspace(0, track_time, track_length)
        return t

