import math

from mongoengine import StringField
import numpy as np
from networks.generators import generator_hurst_exp_network, generator_hurst_exp_network_validation
from networks.network_model import NetworkModel
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from physical_models.models_fbm import FBM
from tracks.simulated_tracks import SimulatedTrack


class HurstExponentNetworkModel(NetworkModel):
    fbm_type = StringField(choices=["Subdiffusive", "Brownian", "Superdiffusive"], required=True)
    model_name = "Hurst Exponent Network"

    net_params = {
        'lr': 1e-4,
        'batch_size': 64,
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
        hurst_exp_keras_model = self.build_model()
        hurst_exp_keras_model.summary()

        callbacks = [
            ModelCheckpoint(filepath="models/{}.h5".format(self.id),
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True)]
        if self.hiperparams_opt:
            validation_generator = generator_hurst_exp_network_validation(batch_size=batch_size,
                                                                          track_length=self.track_length,
                                                                          track_time=self.track_time,
                                                                          fbm_type=self.fbm_type)
        else:
            validation_generator = generator_hurst_exp_network(batch_size=batch_size,
                                                               track_length=self.track_length,
                                                               track_time=self.track_time,
                                                               fbm_type=self.fbm_type)
        history_training = hurst_exp_keras_model.fit(
            x=generator_hurst_exp_network(batch_size=batch_size, track_length=self.track_length,
                                          track_time=self.track_time, fbm_type=self.fbm_type),
            steps_per_epoch=math.ceil(32000 / self.net_params['batch_size']),
            epochs=20,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=math.ceil(8000 / self.net_params['batch_size']))

        self.convert_history_to_db_format(history_training)
        self.keras_model = hurst_exp_keras_model
        if self.hiperparams_opt:
            self.params_training = self.net_params

    def build_model(self):
        inputs = Input(shape=(2, self.track_length))
        x = LSTM(units=64, return_sequences=True, input_shape=(2, self.track_length))(inputs)
        x = LSTM(units=16)(x)
        x = Dense(units=128, activation='selu')(x)
        output_network = Dense(units=1, activation='sigmoid')(x)
        hurst_exp_keras_model = Model(inputs=inputs, outputs=output_network)
        optimizer = Adam(lr=self.net_params['lr'], epsilon=self.net_params['epsilon'],
                         amsgrad=self.net_params['amsgrad'])
        hurst_exp_keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
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
