import math
import pickle

import numpy as np
from keras.utils import to_categorical

from physical_models.models_brownian import Brownian
from physical_models.models_ctrw import CTRW
from physical_models.models_fbm import FBM
from physical_models.models_two_state_obstructed_diffusion import TwoStateObstructedDiffusion


def axis_adaptation_to_net(axis_data, track_length):
    axis_reshaped = np.reshape(axis_data, newshape=[1, len(axis_data)])
    axis_reshaped = axis_reshaped - np.mean(axis_reshaped)
    axis_diff = np.diff(axis_reshaped[0, :track_length])
    return axis_diff


def generate_batch_of_samples_l1(batch_size, track_length, track_time):
    out = np.zeros(shape=[batch_size, track_length - 1, 2])
    label = np.zeros(shape=[batch_size, 1])

    for i in range(batch_size):
        t_sample = np.random.choice(np.linspace(track_time * 0.85, track_time * 1.15, 50))
        track_length_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length * 1.05), 1)))

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
            physical_model = TwoStateObstructedDiffusion.create_random()
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

    for i in range(batch_size):
        t_sample = np.random.choice(np.linspace(track_time * 0.85, track_time * 1.15, 50))
        track_length_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length * 1.05), 1)))
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
        input_net, label = generate_batch_l1_net(batch_size, track_length, track_time)
        yield input_net, label


def generate_batch_l1_net(batch_size, track_length, track_time):
    out, label = generate_batch_of_samples_l1(batch_size=batch_size,
                                              track_length=track_length,
                                              track_time=track_time)
    label = to_categorical(y=label, num_classes=3)
    input_net = np.zeros(shape=[batch_size, track_length - 1, 1])
    for i in range(batch_size):
        input_net[i, :, 0] = out[i, :, 0]
    return input_net, label


# Generator for classification net optimization
def generator_first_layer_validation(batch_size, track_length, track_time, validation_set_size):
    with open('networks/val_data/classification_net/x_val_len_{}_time_{}.pkl'.format(track_length,
                                                                                     track_time),
              'rb') as x_val_data:
        x_val = pickle.load(x_val_data)[0]
    with open('networks/val_data/classification_net/y_val_len_{}_time_{}.pkl'.format(track_length,
                                                                                     track_time),
              'rb') as y_val_data:
        y_val = pickle.load(y_val_data)[0]
    i = 0
    ini = 0
    while True:
        # Generate random data
        if i % 2 == 0:
            input_net, label = generate_batch_l1_net(batch_size, track_length, track_time)
        # Pre-generated dataset
        else:
            input_net = x_val[ini:batch_size]
            label = y_val[ini:batch_size]
        i += 1
        ini += batch_size
        if i >= math.floor(validation_set_size / batch_size):
            i = 0
            ini = 0
        yield input_net, label


def generator_second_layer(batch_size, track_length, track_time):
    while True:
        input_net, label = generate_batch_l2_net(batch_size, track_length, track_time)
        yield input_net, label


def generate_batch_l2_net(batch_size, track_length, track_time):
    out, label = generate_batch_of_samples_l2(batch_size=batch_size,
                                              track_length=track_length,
                                              track_time=track_time)
    label = to_categorical(y=label, num_classes=3)
    input_net = np.zeros(shape=[batch_size, track_length - 1, 1])
    for i in range(batch_size):
        input_net[i, :, 0] = out[i, :, 0]
    return input_net, label


# Generator fbm net for analysis
def generator_second_layer_validation(batch_size, track_length, track_time, validation_set_size):
    with open('networks/val_data/fbm_net/x_val_len_{}_time_{}.pkl'.format(track_length,
                                                                          track_time),
              'rb') as x_val_data:
        x_val = pickle.load(x_val_data)[0]
    with open('networks/val_data/fbm_net/y_val_len_{}_time_{}.pkl'.format(track_length,
                                                                          track_time),
              'rb') as y_val_data:
        y_val = pickle.load(y_val_data)[0]
    i = 0
    ini = 0
    while True:
        # Generate random data
        if i % 2 == 0:
            input_net, label = generate_batch_l2_net(batch_size, track_length, track_time)
        # Pre-generated dataset
        else:
            input_net = x_val[ini:batch_size]
            label = y_val[ini:batch_size]
        i += 1
        ini += batch_size
        if i >= math.floor(validation_set_size / batch_size):
            i = 0
            ini = 0
        yield input_net, label


def generate_batch_of_samples_state_net(batch_size, track_length, track_time):
    out = np.zeros(shape=[batch_size, track_length, 2])
    label = np.zeros(shape=[batch_size, track_length])

    for i in range(batch_size):
        t_sample = np.random.choice(np.linspace(track_time * 0.85, track_time * 1.15, 50))
        track_length_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length * 1.05), 1)))
        model = TwoStateObstructedDiffusion.create_random()
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
        input_net, label = generate_batch_states_net(batch_size, track_length, track_time)
        yield input_net, label


def generate_batch_states_net(batch_size, track_length, track_time):
    out, label = generate_batch_of_samples_state_net(batch_size=batch_size,
                                                     track_length=track_length,
                                                     track_time=track_time)
    input_net = np.zeros(shape=[batch_size, track_length, 1])
    for i in range(batch_size):
        input_net[i, :, 0] = out[i, :, 0]
    return input_net, label


# Generator for states net analysis
def generator_state_net_validation(batch_size, track_length, track_time, validation_set_size):
    with open('networks/val_data/states_net/x_val_len_{}_time_{}.pkl'.format(track_length,
                                                                             track_time),
              'rb') as x_val_data:
        x_val = pickle.load(x_val_data)[0]
    with open('networks/val_data/states_net/y_val_len_{}_time_{}.pkl'.format(track_length,
                                                                             track_time),
              'rb') as y_val_data:
        y_val = pickle.load(y_val_data)[0]
    i = 0
    ini = 0
    while True:
        # Generate random data
        if i % 2 == 0:
            input_net, label = generate_batch_states_net(batch_size, track_length, track_time)
        # Pre-generated dataset
        else:
            input_net = x_val[ini:batch_size]
            label = y_val[ini:batch_size]
        i += 1
        ini += batch_size
        if i >= math.floor(validation_set_size / batch_size):
            i = 0
            ini = 0
        yield input_net, label


def generator_diffusion_coefficient_network(batch_size, track_length, track_time, diffusion_model_range):
    while True:
        out, label = generate_batch_diffusion_coefficient_net(batch_size, diffusion_model_range, track_length,
                                                              track_time)

        yield out, label


def convert_to_diffusion_net_input(x, y):
    r = np.sqrt(x**2 + y**2)
    diff = np.diff(r)
    diff_sq = diff**2
    mean_diff_sq = np.mean(diff_sq)
    diff_sq_norm = diff_sq - mean_diff_sq
    return diff_sq_norm


def generate_batch_diffusion_coefficient_net(track_length, diffusion_model_range, track_time, training_set_size):
    out = np.zeros(shape=[training_set_size, track_length - 1, 1])
    label = np.zeros(shape=[training_set_size, 1])

    for i in range(training_set_size):
        t_sample = np.random.choice(np.linspace(track_time * 0.85, track_time * 1.15, 50))
        track_length_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length * 1.05), 1)))

        if diffusion_model_range == 'Brownian':
            model = Brownian.create_random()
            x_n, y_n, x, y, t = model.simulate_track(track_length_sample, t_sample)
            label[i, 0] = model.get_d_coefficient()
        else:
            model = TwoStateObstructedDiffusion.create_random()
            x_n, y_n, x, y, t = model.simulate_track_only_state0(track_length_sample, t_sample)
            label[i, 0] = model.get_d_state0()

        out[i, :, 0] = convert_to_diffusion_net_input(x_n[:track_length], y_n[:track_length])

    return out, label


# For diffusion coefficient network analysis
def generator_diffusion_coefficient_network_validation(batch_size, track_length, track_time, diffusion_model_range):
    with open('networks/val_data/diffusion_net/x_val_len_{}_time_{}_range_{}.pkl'.format(track_length,
                                                                                         track_time,
                                                                                         diffusion_model_range),
              'rb') as x_val_data:
        x_val = pickle.load(x_val_data)
    with open('networks/val_data/diffusion_net/y_val_len_{}_time_{}_range_{}.pkl'.format(track_length,
                                                                                         track_time,
                                                                                         diffusion_model_range),
              'rb') as y_val_data:
        y_val = pickle.load(y_val_data)
    i = 0

    while True:
        if i % 2 == 0:
            out, label = generate_batch_diffusion_coefficient_net(batch_size, diffusion_model_range, track_length,
                                                                  track_time)
        else:
            out = x_val[i]
            label = y_val[i]
        i += 1
        yield out, label


def generator_hurst_exp_network(batch_size, track_length, track_time, fbm_type):
    while True:
        label, out = generate_batch_hurst_net(batch_size, fbm_type, track_length, track_time)
        yield out, label


def generate_batch_hurst_net(batch_size, fbm_type, track_length, track_time):
    out = np.zeros(shape=(batch_size, 2, track_length))
    label = np.zeros(shape=(batch_size, 1))

    for i in range(batch_size):
        t_sample = np.random.choice(np.linspace(track_time * 0.85, track_time * 1.15, 50))
        track_length_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length * 1.05), 1)))
        if fbm_type == 'Subdiffusive':
            model_sample = FBM.create_random_subdiffusive()
        elif fbm_type == 'Brownian':
            model_sample = FBM.create_random_brownian()
        else:
            model_sample = FBM.create_random_superdiffusive()

        x_noisy, y_nosy, x, y, t = model_sample.simulate_track(track_length=track_length_sample, track_time=t_sample)
        label[i, 0] = model_sample.hurst_exp

        zero_mean_x = x_noisy[:track_length] - np.mean(x_noisy[:track_length])
        zero_mean_x = zero_mean_x / np.std(zero_mean_x)
        out[i, 0, :] = zero_mean_x
        out[i, 1, :] = np.linspace(0, 1, track_length)

    return label, out


def generator_hurst_exp_network_validation(batch_size, track_length, track_time, fbm_type, validation_set_size):
    with open('networks/val_data/hurst_net/x_val_len_{}_time_{}_fbm_type_{}.pkl'.format(track_length,
                                                                                        track_time,
                                                                                        fbm_type),
              'rb') as x_val_data:
        x_val = pickle.load(x_val_data)[0]
    with open('networks/val_data/hurst_net/y_val_len_{}_time_{}_fbm_type_{}.pkl'.format(track_length,
                                                                                        track_time,
                                                                                        fbm_type),
              'rb') as y_val_data:
        y_val = pickle.load(y_val_data)[0]
    i = 0
    ini = 0
    while True:
        # Generate random data
        if i % 2 == 0:
            input_net, label = generate_batch_states_net(batch_size, track_length, track_time)
        # Pre-generated dataset
        else:
            input_net = x_val[ini:batch_size]
            label = y_val[ini:batch_size]
        i += 1
        ini += batch_size
        if i >= math.floor(validation_set_size / batch_size):
            i = 0
            ini = 0
        yield input_net, label
