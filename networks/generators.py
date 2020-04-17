import numpy as np
from keras.utils import to_categorical
from physical_models.models_ctrw import CTRW
from physical_models.models_fbm import FBM
from physical_models.models_two_state_diffusion import TwoStateDiffusion


def axis_adaptation_to_net(axis_data, track_length):
    axis_reshaped = np.reshape(axis_data, newshape=[1, len(axis_data)])
    axis_reshaped = axis_reshaped - np.mean(axis_reshaped)
    axis_diff = np.diff(axis_reshaped[0, :track_length])
    return axis_diff


def generate_batch_of_samples_l1(batch_size, track_length, track_time):
    out = np.zeros(shape=[batch_size, track_length - 1, 2])
    label = np.zeros(shape=[batch_size, 1])
    t_sample = np.random.choice(np.arange(track_time - 2 * (track_time / track_length),
                                          track_time + 2 * (track_time / track_length),
                                          (track_time / (track_length * 2))))
    track_length_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length * 1.05), 1)))

    for i in range(batch_size):
        physical_model_type = np.random.choice(["fbm", "ctrw", "two-state"])
        if physical_model_type == "fbm":
            physical_model = FBM.create_random()
            x_noisy, y_noisy, x, y, t = physical_model.simulate_track(track_length=track_length_sample,
                                                                      track_time=t_sample)
            label[i, 0] = 0
        elif physical_model_type == "ctrw":
            physical_model = CTRW.create_random()
            x_noisy, y_noisy, x, y, t = physical_model.simulate_track(track_length=track_length_sample,
                                                                      track_time=t_sample)
            label[i, 0] = 1
        else:
            physical_model = TwoStateDiffusion.create_random()
            switching = False
            while not switching:
                x_noisy, y_noisy, x, y, t, state, switching = physical_model.simulate_track(
                    track_length=track_length_sample,
                    track_time=t_sample)
            label[i, 0] = 2

        out[i, :, 0] = axis_adaptation_to_net(axis_data=x_noisy, track_length=track_length)
        out[i, :, 1] = axis_adaptation_to_net(axis_data=y_noisy, track_length=track_length)

    return out, label


def generate_batch_of_samples_l2(batch_size, track_length, track_time):
    out = np.zeros(shape=[batch_size, track_length - 1, 2])
    label = np.zeros(shape=[batch_size, 1])
    t_sample = np.random.choice(np.arange(track_time - 2 * (track_time / track_length),
                                          track_time + 2 * (track_time / track_length),
                                          (track_time / (track_length * 2))))
    track_length_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length * 1.05), 1)))

    for i in range(batch_size):
        model_sample = np.random.choice(["sub", "brownian", "super"])
        if model_sample == "sub":
            model = FBM.create_random_subdiffusive()
            label[i, 0] = 0

        elif model_sample == "brownian":
            model = FBM.create_random_brownian()
            label[i, 0] = 1

        else:
            model = FBM.create_random_superdiffusive()
            label[i, 0] = 2

        x_noisy, y_noisy, x, y, t = model.simulate_track(track_length=track_length_sample, track_time=t_sample)

        out[i, :, 0] = axis_adaptation_to_net(axis_data=x_noisy, track_length=track_length)
        out[i, :, 1] = axis_adaptation_to_net(axis_data=y_noisy, track_length=track_length)

    return out, label


def generator_first_layer(batch_size, track_length, track_time):
    while True:
        out, label = generate_batch_of_samples_l1(batch_size=batch_size,
                                                  track_length=track_length,
                                                  track_time=track_time)
        label = to_categorical(y=label, num_classes=3)
        input_net = np.zeros(shape=[batch_size, track_length - 1, 1])

        for i in range(batch_size):
            input_net[i, :, 0] = out[i, :, 0]

        yield input_net, label


def generator_second_layer(batch_size, track_length, track_time):
    while True:
        out, label = generate_batch_of_samples_l2(batch_size=batch_size,
                                                  track_length=track_length,
                                                  track_time=track_time)
        label = to_categorical(y=label, num_classes=3)
        input_net = np.zeros(shape=[batch_size, track_length - 1, 1])
        for i in range(batch_size):
            input_net[i, :, 0] = out[i, :, 0]
        yield input_net, label


def generate_batch_of_samples_state_net(batch_size, track_length, track_time):
    out = np.zeros(shape=[batch_size, track_length, 2])
    label = np.zeros(shape=[batch_size, track_length])
    t_sample = np.random.choice(np.arange(track_time - 2 * (track_time / track_length),
                                          track_time + 2 * (track_time / track_length),
                                          (track_time / (track_length * 2))))
    track_length_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length * 1.05), 1)))

    for i in range(batch_size):
        model = TwoStateDiffusion.create_random()
        switching = False
        while not switching:
            x_noisy, y_noisy, x, y, t, state, switching = model.simulate_track(track_length=track_length_sample,
                                                                               track_time=t_sample)

        axis_reshaped = np.reshape(x_noisy, [1, len(x_noisy)])[:, :track_length]
        out[i, :, 0] = axis_reshaped - np.mean(axis_reshaped)

        axis_reshaped = np.reshape(y_noisy, [1, len(y_noisy)])[:, :track_length]
        out[i, :, 1] = axis_reshaped - np.mean(axis_reshaped)

        label[i, :] = state[:track_length]

    return out, label


def generator_state_net(batch_size, track_length, track_time):
    while True:
        out, label = generate_batch_of_samples_state_net(batch_size=batch_size,
                                                         track_length=track_length,
                                                         track_time=track_time)

        input_net = np.zeros(shape=[batch_size, track_length, 1])
        for i in range(batch_size):
            input_net[i, :, 0] = out[i, :, 0]

        yield input_net, label


def generator_diffusion_coefficient_network(batch_size, track_length, track_time, state, denoising_model):
    assert (state == 0 or state == 1), "State must be 0 or 1"
    assert denoising_model.diffusion_model_state == state, "Invalid state denoising model"
    while True:
        t_sample = np.random.choice(np.arange(track_time - 2 * (track_time / track_length),
                                              track_time + 2 * (track_time / track_length),
                                              (track_time / (track_length * 2))))
        out = np.zeros(shape=[batch_size, 2, 1])
        label = np.zeros(shape=[batch_size, 1])
        noisy_out = np.zeros(shape=[batch_size, track_length])
        m_noisy_out = np.zeros(shape=batch_size)

        for i in range(batch_size):

            two_state_model = TwoStateDiffusion.create_random()
            if state == 0:
                x_noisy, y_noisy, x, y, t = two_state_model.simulate_track_only_state0(track_length=track_length,
                                                                                       track_time=t_sample)
                label[i, 0] = two_state_model.normalize_d_coefficient_to_net(state_number=0)
            else:
                x_noisy, y_noisy, x, y, t = two_state_model.simulate_track_only_state1(track_length=track_length,
                                                                                       track_time=t_sample)
                label[i, 0] = two_state_model.normalize_d_coefficient_to_net(state_number=1)

            m_noisy_out[i] = np.mean(x_noisy)
            noisy_out[i, :] = x_noisy - m_noisy_out[i]

        if state == 1:
            in_net = np.zeros(shape=[batch_size, track_length, 1])
            in_net[:, :, 0] = noisy_out
            noise_reduced_x = denoising_model.keras_model.predict(in_net)
            for j in range(batch_size):
                noise_reduced_x[j, :] = m_noisy_out[j] * np.ones(shape=track_length)
        else:
            noise_reduced_x = noisy_out  # Reducing noise does not make a major improvement using state 0

        dx = np.diff(noise_reduced_x, axis=1)
        m = np.mean(np.abs(dx), axis=1)
        s = np.std(dx, axis=1)

        for i in range(batch_size):
            out[i, :, 0] = [m[i], s[i]]

        yield out, label


def generator_noise_reduction_net(batch_size, track_length, track_time, diffusion_model_state):
    assert (diffusion_model_state == 0 or diffusion_model_state == 1), "State must be 0 or 1"

    while True:
        t_sample = np.random.choice(np.arange(track_time - 2 * (track_time / track_length),
                                              track_time + 2 * (track_time / track_length),
                                              (track_time / (track_length * 2))))

        out = np.zeros(shape=[batch_size, track_length, 1])
        label = np.zeros(shape=[batch_size, track_length])

        for i in range(batch_size):

            two_state_model = TwoStateDiffusion.create_random()
            if diffusion_model_state == 0:
                x_noisy, y_noisy, x, y, t = two_state_model.simulate_track_only_state0(track_length=track_length,
                                                                                       track_time=t_sample)

            else:
                x_noisy, y_noisy, x, y, t = two_state_model.simulate_track_only_state1(track_length=track_length,
                                                                                       track_time=t_sample)

            out[i, :, 0] = x_noisy - np.mean(x_noisy)
            label[i, :] = x - np.mean(x_noisy)

        yield out, label
