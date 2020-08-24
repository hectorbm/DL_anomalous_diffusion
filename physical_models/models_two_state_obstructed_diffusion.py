import numpy as np
from . import models
from . import models_noise


class TwoStateObstructedDiffusion(models.Models):
    """
    State-0: Free Diffusion
    State-1: Obstructed Diffusion
    """
    d0_low = 0.05
    d0_high = 0.2
    k0_low = 0.01
    k0_high = 0.08
    k1_low = 0.007
    k1_high = 0.2

    def __init__(self, k_state0, k_state1, d_state0):
        self.k_state0 = k_state0
        self.k_state1 = k_state1
        self.D_state0 = d_state0 * 1000000  # Convert from um^2 -> nm^2

    @classmethod
    def create_random(cls):
        # k_state(i) dimensions = 1 / frame
        # D_state(i) dimensions = um^2 * s^(-beta)
        d_state0 = np.random.uniform(low=cls.d0_low, high=cls.d0_high)
        k_state0 = np.random.uniform(low=cls.k0_low, high=cls.k0_high)
        k_state1 = np.random.uniform(low=cls.k1_low, high=cls.k1_high)
        model = cls(k_state0, k_state1, d_state0)
        return model

    @classmethod
    def create_with_coefficients(cls, k_state0, k_state1, d_state0):
        assert (d_state0 > 0), "Invalid Diffusion coefficient state-0"
        assert (k_state0 > 0), "Invalid switching rate state-0"
        assert (k_state0 > 0), "Invalid switching rate state-1"
        return cls(k_state0, k_state1, d_state0)

    def get_d_state0(self):
        return self.D_state0 / 1000000

    def normalize_d_coefficient_to_net(self, state_number):
        assert (state_number == 0), "Not a valid state"
        delta_d0 = self.d0_high - self.d0_low
        return (1 / delta_d0) * (self.get_d_state0() - self.d0_low)

    @classmethod
    def denormalize_d_coefficient_to_net(cls, output_coefficient_net):
        delta_d0 = cls.d0_high - cls.d0_low
        return output_coefficient_net * delta_d0 + cls.d0_low

    def simulate_track(self, track_length, track_time):
        x = np.zeros(shape=track_length)
        y = np.zeros(shape=track_length)

        state, switching = self.simulate_switching_states(track_length)

        if state[0] == 1:
            x[0] = np.random.normal(loc=0, scale=5)
            y[0] = np.random.normal(loc=0, scale=5)
        else:
            x[0] = np.random.normal(loc=0, scale=1)
            y[0] = np.random.normal(loc=0, scale=1)

        i = 1
        while i < track_length:
            while state[i] == 0 and i < track_length:
                jumps_x = np.random.normal(loc=0, scale=1)
                jumps_y = np.random.normal(loc=0, scale=1)
                x[i] = x[i - 1] + (jumps_x * np.sqrt(2 * self.D_state0 * (track_time / track_length)))
                y[i] = y[i - 1] + (jumps_y * np.sqrt(2 * self.D_state0 * (track_time / track_length)))
                i += 1
                if i >= track_length:
                    break

            confinement_flag = True
            if i >= track_length:
                break
            while state[i] == 1 and i < track_length:
                if confinement_flag:
                    confinement_region_min_x, confinement_region_max_x = self.simulate_confinement_region(x[i - 1])
                    confinement_region_min_y, confinement_region_max_y = self.simulate_confinement_region(y[i - 1])
                    confinement_flag = False

                jumps_x = np.random.normal(loc=0, scale=5)
                jumps_y = np.random.normal(loc=0, scale=5)

                if jumps_x > 0:
                    if x[i - 1] + jumps_x > confinement_region_max_x:
                        x[i] = x[i - 1]
                    else:
                        x[i] = x[i - 1] + jumps_x
                else:
                    if x[i - 1] + jumps_x < confinement_region_min_x:
                        x[i] = x[i - 1]
                    else:
                        x[i] = x[i - 1] + jumps_x

                if jumps_y > 0:
                    if y[i - 1] + jumps_y > confinement_region_max_y:
                        y[i] = y[i - 1]
                    else:
                        y[i] = y[i - 1] + jumps_y

                else:
                    if y[i - 1] + jumps_y < confinement_region_min_y:
                        y[i] = y[i - 1]
                    else:
                        y[i] = y[i - 1] + jumps_y

                i += 1
                if i >= track_length:
                    break

        x, x_noisy, y, y_noisy = models_noise.add_noise_and_offset(track_length, x, y)

        t = self.simulate_tract_time(track_length, track_time)

        return x_noisy, y_noisy, x, y, t, state, switching

    def simulate_track_only_state0(self, track_length, track_time):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        for i in range(track_length):
            x[i] = x[i] * np.sqrt(2 * self.D_state0 * (track_time / track_length))
            y[i] = y[i] * np.sqrt(2 * self.D_state0 * (track_time / track_length))

        x = np.cumsum(x)
        y = np.cumsum(y)

        x, x_noisy, y, y_noisy = models_noise.add_noise_and_offset(track_length, x, y)

        t = self.simulate_tract_time(track_length, track_time)

        return x_noisy, y_noisy, x, y, t

    def simulate_track_only_state1(self, track_length, track_time):
        initial_pos_x = np.random.normal(loc=0, scale=5)
        initial_pos_y = np.random.normal(loc=0, scale=5)

        confinement_region_min_x, confinement_region_max_x = self.simulate_confinement_region(initial_pos_x)
        confinement_region_min_y, confinement_region_max_y = self.simulate_confinement_region(initial_pos_y)

        x = np.zeros(shape=track_length)
        y = np.zeros(shape=track_length)

        jumps_x = np.random.normal(loc=0, scale=5, size=track_length)
        jumps_y = np.random.normal(loc=0, scale=5, size=track_length)

        x[0], y[0] = initial_pos_x, initial_pos_y

        for i in range(1, track_length):
            if jumps_x[i] > 0:
                if x[i - 1] + jumps_x[i] > confinement_region_max_x:
                    x[i] = x[i - 1]
                else:
                    x[i] = x[i - 1] + jumps_x[i]

            else:
                if x[i - 1] + jumps_x[i] < confinement_region_min_x:
                    x[i] = x[i - 1]
                else:
                    x[i] = x[i - 1] + jumps_x[i]

            if jumps_y[i] > 0:
                if y[i - 1] + jumps_y[i] > confinement_region_max_y:
                    y[i] = y[i - 1]
                else:
                    y[i] = y[i - 1] + jumps_y[i]

            else:
                if y[i - 1] + jumps_y[i] < confinement_region_min_y:
                    y[i] = y[i - 1]
                else:
                    y[i] = y[i - 1] + jumps_y[i]

        x, x_noisy, y, y_noisy = models_noise.add_noise_and_offset(track_length, x, y)

        t = self.simulate_tract_time(track_length, track_time)

        return x_noisy, y_noisy, x, y, t

    def simulate_tract_time(self, track_length, track_time):
        t = np.linspace(0, track_time, track_length)
        return t

    def simulate_confinement_region(self, initial_pos):
        confinement_region_size = np.random.uniform(low=20, high=40)
        offset_region = initial_pos + np.random.uniform(low=(-confinement_region_size / 2),
                                                        high=(confinement_region_size / 2))

        confinement_region_max = offset_region + (confinement_region_size / 2)
        confinement_region_min = offset_region - (confinement_region_size / 2)

        return confinement_region_min, confinement_region_max

    def simulate_switching_states(self, track_length):
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
