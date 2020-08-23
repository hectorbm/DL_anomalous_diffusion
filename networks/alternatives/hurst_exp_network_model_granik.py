from mongoengine import StringField
import numpy as np
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Conv1D
from keras.layers import Input, GlobalMaxPooling1D, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from networks.generators import generator_hurst_exp_network_granik
from networks.network_model import NetworkModel
from physical_models.models_fbm import FBM
from tracks.simulated_tracks import SimulatedTrack


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[np.int(result.size / 2):]


class HurstExponentNetworkModelGranik(NetworkModel):
    fbm_type = StringField(choices=["subdiffusive", "superdiffusive"], required=True)
    model_name = "Hurst Exponent Network (Granik et. al.)"

    def train_network(self, batch_size):
        initializer = 'he_normal'
        f = 32  # number of convolution filters in a single network layer

        inputs = Input((self.track_length - 1, 1))

        x1 = Conv1D(f, 4, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 4, dilation_rate=2, padding='causal', activation='relu', kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(f, 4, dilation_rate=4, padding='causal', activation='relu', kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)
        x1 = GlobalMaxPooling1D()(x1)

        x2 = Conv1D(f, 2, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 2, dilation_rate=2, padding='causal', activation='relu', kernel_initializer=initializer)(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(f, 2, dilation_rate=4, padding='causal', activation='relu', kernel_initializer=initializer)(x2)
        x2 = BatchNormalization()(x2)
        x2 = GlobalMaxPooling1D()(x2)

        x3 = Conv1D(f, 3, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 3, dilation_rate=2, padding='causal', activation='relu', kernel_initializer=initializer)(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(f, 3, dilation_rate=4, padding='causal', activation='relu', kernel_initializer=initializer)(x3)
        x3 = BatchNormalization()(x3)
        x3 = GlobalMaxPooling1D()(x3)

        x4 = Conv1D(f, 10, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=5, padding='causal', activation='relu', kernel_initializer=initializer)(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(f, 10, dilation_rate=10, padding='causal', activation='relu', kernel_initializer=initializer)(x4)
        x4 = BatchNormalization()(x4)
        x4 = GlobalMaxPooling1D()(x4)

        con = concatenate([x1, x2, x3, x4])
        dense = Dense(512, activation='relu')(con)
        dense = Dense(256, activation='relu')(dense)
        dense2 = Dense(1, activation='sigmoid')(dense)
        hurst_exp_keras_model_granik = Model(inputs=inputs, outputs=dense2)

        optimizer = Adam(lr=1e-5)
        hurst_exp_keras_model_granik.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        hurst_exp_keras_model_granik.summary()

        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=20,
                                   verbose=1,
                                   min_delta=1e-4),
                     ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=4,
                                       verbose=1,
                                       min_lr=1e-12),
                     ModelCheckpoint(filepath="models/{}.h5".format(self.id),
                                     monitor='val_loss',
                                     save_best_only=True,
                                     mode='min',
                                     save_weights_only=False)]

        history_training = hurst_exp_keras_model_granik.fit_generator(
            generator=generator_hurst_exp_network_granik(batch_size=batch_size,
                                                         track_length=self.track_length,
                                                         track_time=self.track_time,
                                                         fbm_type=self.fbm_type),
            steps_per_epoch=500,
            epochs=15,
            callbacks=callbacks,
            validation_data=generator_hurst_exp_network_granik(batch_size=batch_size,
                                                               track_length=self.track_length,
                                                               track_time=self.track_time,
                                                               fbm_type=self.fbm_type),
            validation_steps=50)

        self.convert_history_to_db_format(history_training)
        self.keras_model = hurst_exp_keras_model_granik

    def evaluate_track_input(self, track):
        assert (track.track_length == self.track_length), "Invalid track length"
        prediction = np.zeros(shape=track.n_axes)
        out = np.zeros(shape=(1, self.track_length-1, 1))

        for i in range(track.n_axes):
            dx = np.diff(track.axes_data[str(i)], axis=0)
            out[0, :, 0] = autocorr((dx - np.mean(dx)) / (np.std(dx)))
            prediction[i] = self.keras_model.predict(out)

        return np.mean(prediction)

    def validate_test_data_mse(self, n_axes):
        test_batch_size = 100
        mse = np.zeros(shape=test_batch_size)

        for i in range(test_batch_size):
            if self.fbm_type == "subdiffusive":
                model_sample = FBM.create_random_subdiffusive()
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
