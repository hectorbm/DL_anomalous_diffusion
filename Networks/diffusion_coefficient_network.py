import math

from sklearn.metrics import mean_squared_error

from PhysicalModels.brownian import Brownian
from Tracks.simulated_tracks import SimulatedTrack
from . import network

from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from Networks.generators import generator_diffusion_coefficient_network, \
    generator_diffusion_coefficient_network_validation, generate_batch_diffusion_coefficient_net, \
    convert_to_diffusion_net_input
from PhysicalModels.two_states_obstructed_diffusion import TwoStateObstructedDiffusion
from mongoengine import StringField
import numpy as np


class DiffusionCoefficientNetworkModel(network.NetworkModel):
    diffusion_model_range = StringField(choices=["2-State-OD", "Brownian"])

    net_params = {
        'training_set_size': 50000,
        'validation_set_size': 12500,
        'lr': 1e-7,
        'batch_size': 8,
        'amsgrad': False,
        'epsilon': 1e-7
    }
    # For analysis of hyper-params
    analysis_params = {
        'lr': [1e-7, 2e-7, 9e-8, 3e-7],
        'amsgrad': [False, True],
        'batch_size': [8, 16, 32],
        'epsilon': [1e-6, 1e-7, 1e-8]
    }

    def train_network(self, batch_size):
        x_data, y_data = generate_batch_diffusion_coefficient_net(self.track_length,
                                                                  self.diffusion_model_range,
                                                                  self.track_time,
                                                                  self.net_params['training_set_size'])

        diffusion_coefficient_keras_model = self.build_model()
        diffusion_coefficient_keras_model.summary()

        if self.hiperparams_opt:
            validation_generator = generator_diffusion_coefficient_network_validation(batch_size,
                                                                                      self.track_length,
                                                                                      self.track_time,
                                                                                      self.diffusion_model_range)
        else:
            validation_generator = generator_diffusion_coefficient_network(batch_size,
                                                                           self.track_length,
                                                                           self.track_time,
                                                                           self.diffusion_model_range)
        history_training = diffusion_coefficient_keras_model.fit(
            x=x_data,
            y=y_data,
            epochs=50,
            batch_size=self.net_params['batch_size'],
            validation_data=validation_generator,
            validation_steps=math.floor(self.net_params['validation_set_size'] / self.net_params['batch_size']),
            shuffle=True
        )

        self.convert_history_to_db_format(history_training)
        self.keras_model = diffusion_coefficient_keras_model
        self.keras_model.save('Models/{}'.format(self.id))

        if self.hiperparams_opt:
            self.params_training = self.net_params

    def build_model(self):
        filters = 32

        inputs = Input(shape=(self.track_length - 1, 1))
        x1 = Conv1D(filters=filters, kernel_size=2,
                    padding='causal',
                    activation='relu',
                    kernel_initializer='he_normal')(inputs)

        x1 = BatchNormalization()(x1)
        x1 = GlobalMaxPooling1D()(x1)

        x = Dense(units=256, activation='relu')(x1)
        x = Dense(units=128, activation='relu')(x)
        output = Dense(units=1, activation='relu')(x)

        model = Model(inputs=inputs, outputs=output)

        optimizer = Adam(lr=self.net_params['lr'], amsgrad=self.net_params['amsgrad'], epsilon=self.net_params['epsilon'])

        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

        return model

    def evaluate_track_input(self, track):
        if track.track_length != self.track_length:
            raise ValueError('Invalid track length')

        axes_data = track.axes_data
        out = np.zeros(shape=[1, self.track_length - 1, 1])
        out[0, :, 0] = convert_to_diffusion_net_input(axes_data['0'], axes_data['1'])

        if self.diffusion_model_range == "2-State-OD":
            prediction = self.keras_model.predict(out)
        else:
            prediction = self.keras_model.predict(out)

        return prediction[0, 0]

    def validate_test_data_mse(self, n_axes, test_batch_size=100):
        mse_avg = np.zeros(shape=test_batch_size)
        for i in range(test_batch_size):

            if self.diffusion_model_range == "2-State-OD":
                model = TwoStateObstructedDiffusion.create_random()

                x_noisy, y_noisy, x, y, t = model.simulate_track_only_state0(track_length=self.track_length,
                                                                             track_time=self.track_time)
                ground_truth = model.get_d_state0()

            else:
                model = Brownian.create_random()
                x_noisy, y_noisy, x, y, t = model.simulate_track(track_length=self.track_length,
                                                                 track_time=self.track_time)
                ground_truth = model.get_d_coefficient()

            noisy_data = [x_noisy, y_noisy]
            track = SimulatedTrack(track_length=self.track_length, track_time=self.track_time,
                                   n_axes=n_axes, model_type=model.__class__.__name__)
            track.set_axes_data(axes_data=noisy_data)
            track.set_time_axis(time_axis_data=t)
            predicted_coefficient = self.evaluate_track_input(track)
            mse_avg[i] = mean_squared_error(y_true=[ground_truth], y_pred=[predicted_coefficient])

        return np.mean(mse_avg)
