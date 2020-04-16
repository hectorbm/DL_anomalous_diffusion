from . import network_model
from mongoengine import IntField
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Conv1D, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from network_models.generators import generator_noise_reduction_net
import numpy as np


class NoiseReductionNetworkModel(network_model.NetworkModel):
    diffusion_model_state = IntField(choices=[0, 1], required=True)

    def train_network(self, batch_size):
        initializer = 'he_normal'
        filters_size = 20
        kernel_size = 2

        inputs = Input(shape=(self.track_length, 1))
        x = Conv1D(filters=filters_size, kernel_size=kernel_size, padding='causal', activation='relu',
                   kernel_initializer=initializer)(inputs)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        dense_1 = Dense(units=self.track_length, activation='relu')(x)
        output_network = Dense(units=self.track_length)(dense_1)

        noise_reduction_keras_model = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=1e-3)
        noise_reduction_keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
        noise_reduction_keras_model.summary()

        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=20,
                                   verbose=1,
                                   min_delta=1e-4),
                     ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=4,
                                       verbose=1,
                                       min_lr=1e-9),
                     ModelCheckpoint(filepath="models/{}.h5".format(self.id),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True)]

        history_training = noise_reduction_keras_model.fit(
            x=generator_noise_reduction_net(batch_size=batch_size,
                                            track_length=self.track_length,
                                            track_time=self.track_time,
                                            diffusion_model_state=self.diffusion_model_state),
            steps_per_epoch=1000,
            epochs=3,
            callbacks=callbacks,
            validation_data=generator_noise_reduction_net(batch_size=batch_size,
                                                          track_length=self.track_length,
                                                          track_time=self.track_time,
                                                          diffusion_model_state=self.diffusion_model_state),
            validation_steps=100)

        self.keras_model = noise_reduction_keras_model
        self.convert_history_to_db_format(history_training)

    def evaluate_track_input(self, track):
        assert track.track_length == self.track_length, "Invalid track length"

        if self.keras_model is None:
            self.load_model_from_file()

        model_predictions = np.zeros(shape=(track.n_axes, self.track_length))

        for axis in range(track.n_axes):
            input_net = np.zeros(shape=[1, self.track_length, 1])
            m_noisy = np.mean(track.axes_data[str(axis)])
            input_net[0, :, 0] = track.axes_data[str(axis)] - m_noisy
            model_predictions[axis, :] = (self.keras_model.predict(input_net)[0, :]) + m_noisy

        return model_predictions

    def validate_test_data_mse(self, n_axes):
        pass
