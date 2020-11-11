import math

from tensorflow.keras.callbacks import EarlyStopping
from mongoengine import StringField
import numpy as np
from Networks.generators import generator_hurst_exp_network, generator_hurst_exp_network_validation, \
    generate_batch_hurst_net
from Networks.network import NetworkModel
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from PhysicalModels.fbm import FBM
from Tracks.simulated_tracks import SimulatedTrack



class HurstExponentNetworkModel(NetworkModel):
    fbm_type = StringField(choices=["Subdiffusive", "Brownian", "Superdiffusive"], required=True)
    model_name = "Hurst Exponent Network"

    net_params = {
        'training_set_size': 40000,
        'validation_set_size': 10000,
        'Subdiffusive': {
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': True,
            'epsilon': 1e-6
        },
        'Brownian': {
            'lr': 0.0001,
            'batch_size': 16,
            'amsgrad': False,
            'epsilon': 1e-8
        },
        'Superdiffusive': {
            'lr': 0.001,
            'batch_size': 64,
            'amsgrad': False,
            'epsilon': 1e-6
        }
    }
    # For analysis of hyper-params
    analysis_params = {
        'lr': [1e-2, 1e-3, 1e-4, 1e-5],
        'amsgrad': [False, True],
        'batch_size': [16, 32, 64],
        'epsilon': [1e-6, 1e-7, 1e-8]
    }

    def train_network(self, batch_size):
        y_data, x_data = generate_batch_hurst_net(self.net_params['training_set_size'],
                                                  self.fbm_type,
                                                  self.track_length,
                                                  self.track_time)

        hurst_exp_keras_model = self.build_model()
        hurst_exp_keras_model.summary()

        callbacks = [EarlyStopping(monitor="val_loss",
                                   min_delta=1e-4,
                                   patience=5,
                                   verbose=1,
                                   mode="min")]

        if self.hiperparams_opt:
            validation_generator = generator_hurst_exp_network_validation(batch_size=self.net_params[self.fbm_type]['batch_size'],
                                                                          track_length=self.track_length,
                                                                          track_time=self.track_time,
                                                                          fbm_type=self.fbm_type,
                                                                          validation_set_size=self.net_params['validation_set_size'])
        else:
            validation_generator = generator_hurst_exp_network(batch_size=self.net_params[self.fbm_type]['batch_size'],
                                                               track_length=self.track_length,
                                                               track_time=self.track_time,
                                                               fbm_type=self.fbm_type)
        history_training = hurst_exp_keras_model.fit(
            x=x_data,
            y=y_data,
            epochs=20,
            batch_size=self.net_params['batch_size'],
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=math.floor(self.net_params['validation_set_size'] / self.net_params[self.fbm_type]['batch_size']),
            shuffle=True)

        self.convert_history_to_db_format(history_training)
        self.keras_model = hurst_exp_keras_model
        self.keras_model.save(filepath="Models/{}".format(self.id))

        if self.hiperparams_opt:
            self.params_training = self.net_params

    def build_model(self):
        inputs = Input(shape=(2, self.track_length))
        x = LSTM(units=64, return_sequences=True, input_shape=(2, self.track_length))(inputs)
        x = LSTM(units=16)(x)
        x = Dense(units=128, activation='selu')(x)
        output_network = Dense(units=1, activation='sigmoid')(x)

        hurst_exp_keras_model = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=self.net_params[self.fbm_type]['lr'], epsilon=self.net_params[self.fbm_type]['epsilon'],
                         amsgrad=self.net_params[self.fbm_type]['amsgrad'])

        hurst_exp_keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

        return hurst_exp_keras_model

    def evaluate_track_input(self, track):
        assert (track.track_length == self.track_length), "Invalid track length"
        prediction = np.zeros(shape=track.n_axes)
        out = np.zeros(shape=(1, 2, self.track_length))

        for i in range(track.n_axes):
            zero_mean_x = track.axes_data[str(i)] - np.mean(track.axes_data[str(i)])
            zero_mean_x = zero_mean_x / np.std(zero_mean_x)
            out[0, 0, :] = zero_mean_x
            out[0, 1, :] = np.linspace(0, 1, self.track_length)
            prediction[i] = self.keras_model.predict(out)[0, 0]

        return np.mean(prediction)

    def validate_test_data_mse(self, n_axes):
        test_batch_size = 100
        mse = np.zeros(shape=test_batch_size)

        for i in range(test_batch_size):
            if self.fbm_type == "Subdiffusive":
                model_sample = FBM.create_random_subdiffusive()
            elif self.fbm_type == 'Brownian':
                model_sample = FBM.create_random_brownian()
            else:
                model_sample = FBM.create_random_superdiffusive()

            x_noisy, y_noisy, x, y, t = model_sample.simulate_track(track_length=self.track_length,
                                                                    track_time=self.track_time)
            ground_truth = model_sample.hurst_exp
            noisy_data = [x_noisy, y_noisy]
            track = SimulatedTrack(track_length=self.track_length, track_time=self.track_time,
                                   n_axes=n_axes, model_type=model_sample.__class__.__name__)
            track.set_axes_data(axes_data=noisy_data)
            track.set_time_axis(time_axis_data=t)
            prediction = self.evaluate_track_input(track)
            mse[i] = mean_squared_error(ground_truth, prediction)

        return np.mean(mse)
