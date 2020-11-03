import math

from sklearn.metrics import mean_squared_error

from physical_models.models_brownian import Brownian
from tracks.simulated_tracks import SimulatedTrack
from . import network_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from networks.generators import generator_diffusion_coefficient_network, \
    generator_diffusion_coefficient_network_validation
from physical_models.models_two_state_obstructed_diffusion import TwoStateObstructedDiffusion
from mongoengine import StringField
import numpy as np


class DiffusionCoefficientNetworkModel(network_model.NetworkModel):
    diffusion_model_range = StringField(choices=["2-State-OD", "Brownian"])

    net_params = {
        'lr': 1e-3,
        'batch_size': 8,
        'amsgrad': False,
        'epsilon': 1e-7
    }
    # For analysis of hyper-params
    analysis_params = {
        'lr': [1e-2, 1e-3, 1e-4, 1e-5],
        'amsgrad': [False, True],
        'batch_size': [8, 16, 32],
        'epsilon': [1e-6, 1e-7, 1e-8]
    }

    def train_network(self, batch_size):
        diffusion_coefficient_keras_model = self.build_model()
        diffusion_coefficient_keras_model.summary()

        callbacks = [
            ModelCheckpoint(filepath="models/{}.h5".format(self.id),
                            monitor='val_loss',
                            save_best_only=True,
                            verbose=1)]

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
            x=generator_diffusion_coefficient_network(batch_size,
                                                      self.track_length,
                                                      self.track_time,
                                                      self.diffusion_model_range),
            steps_per_epoch=math.ceil(19200 / self.net_params['batch_size']),
            epochs=15,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=math.ceil(4800 / self.net_params['batch_size']))
        self.convert_history_to_db_format(history_training)
        self.keras_model = diffusion_coefficient_keras_model
        if self.hiperparams_opt:
            self.params_training = self.net_params

    def build_model(self):
        initializer = 'he_normal'
        filters_size = 32
        x_kernel_size = 2
        inputs = Input(shape=(2, 1))
        x = Conv1D(filters=filters_size, kernel_size=x_kernel_size, padding='same', activation='relu',
                   kernel_initializer=initializer)(inputs)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling1D()(x)
        dense_1 = Dense(units=512, activation='relu')(x)
        dense_2 = Dense(units=256, activation='relu')(dense_1)
        output_network = Dense(units=1, activation='sigmoid')(dense_2)
        diffusion_coefficient_keras_model = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=self.net_params['lr'],
                         epsilon=self.net_params['epsilon'],
                         amsgrad=self.net_params['amsgrad'])

        diffusion_coefficient_keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
        return diffusion_coefficient_keras_model

    def evaluate_track_input(self, track):
        assert track.track_length == self.track_length, "Invalid track length"
        prediction = np.zeros(shape=track.n_axes)
        out = np.zeros(shape=(1, 2, 1))

        axes_data = track.axes_data

        for axis in range(track.n_axes):
            d = np.diff(axes_data[str(axis)], axis=0)
            m = np.mean(np.abs(d), axis=0)
            s = np.std(d, axis=0)
            out[0, :, 0] = [m, s]
            prediction[axis] = self.keras_model.predict(out[:, :, :])

        if self.diffusion_model_range == "2-State-OD":
            mean_prediction = TwoStateObstructedDiffusion.denormalize_d_coefficient_to_net(
                output_coefficient_net=np.mean(prediction))
        else:
            mean_prediction = Brownian.denormalize_d_coefficient_to_net(
                output_coefficient_net=np.mean(prediction))

        return mean_prediction

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
