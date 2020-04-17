from physical_models.models_two_state_diffusion import TwoStateDiffusion
from tracks.simulated_tracks import SimulatedTrack
from . import network_model
from mongoengine import IntField
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Conv1D, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from networks.generators import generator_noise_reduction_net
import numpy as np
from sklearn.metrics import mean_squared_error


class NoiseReductionNetworkModel(network_model.NetworkModel):
    diffusion_model_state = IntField(choices=[0, 1], required=True)
    model_name = 'Noise Reduction Network'

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
            epochs=10,
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

    def validate_test_data_mse(self, n_axes, test_batch_size=100):
        mse_avg = np.zeros(shape=test_batch_size)
        for i in range(test_batch_size):
            two_state_model = TwoStateDiffusion.create_random()
            if self.diffusion_model_state == 0:
                x_noisy, y_noisy, x, y, t = two_state_model.simulate_track_only_state0(track_length=self.track_length,
                                                                                       track_time=self.track_time)

            else:
                x_noisy, y_noisy, x, y, t = two_state_model.simulate_track_only_state1(track_length=self.track_length,
                                                                                       track_time=self.track_time)
            noisy_data = [x_noisy, y_noisy]
            ground_truth = [x, y]
            track = SimulatedTrack(track_length=self.track_length, track_time=self.track_time,
                                   n_axes=n_axes, model_type=two_state_model.__class__.__name__)
            track.set_axes_data(axes_data=noisy_data)
            track.set_time_axis(time_axis_data=t)
            denoised_axes_data = self.evaluate_track_input(track)

            mse_axis = np.zeros(shape=track.n_axes)
            for axis in range(track.n_axes):
                mse_axis[axis] = mean_squared_error(y_true=ground_truth[axis], y_pred=denoised_axes_data[axis])
            mse_avg[i] = np.mean(mse_axis)

        return np.mean(mse_avg)

