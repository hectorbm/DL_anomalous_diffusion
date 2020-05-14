from networks import network_model
from networks.generators import generator_first_layer_spectrum, axis_adaptation_to_net_spectrum
from physical_models.models_ctrw import CTRW
from physical_models.models_fbm import FBM
import numpy as np
from keras.layers import Conv2D, GlobalMaxPooling2D
from keras.layers import Dense, Input, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from physical_models.models_two_state_diffusion import TwoStateDiffusion
from tools.analysis_tools import plot_confusion_matrix_for_layer
from tracks.simulated_tracks import SimulatedTrack


class L1NetworkSpectrumModel(network_model.NetworkModel):
    output_categories = 3
    output_categories_labels = ["fBm", "CTRW", "2-State"]
    model_name = 'L1 Network(Spectrum)'

    def train_network(self, batch_size):
        initializer = 'he_normal'

        inputs = Input(shape=(2, self.track_length, 1))
        x1 = Conv2D(filters=127, kernel_size=(2, 2), activation='relu', padding='same',
                    kernel_initializer=initializer)(inputs)
        x1 = BatchNormalization()(x1)

        x1 = Conv2D(filters=127, kernel_size=(2, 2), activation='relu', padding='same',
                    kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)

        x1 = Conv2D(filters=127, kernel_size=(2, 2), activation='relu', padding='same',
                    kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)
        x1 = GlobalMaxPooling2D()(x1)

        x1 = Dense(units=256, activation='selu')(x1)
        x1 = Dense(units=64, activation='selu')(x1)
        output_network = Dense(units=3, activation='softmax')(x1)
        l2_spectrum_keras_model = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=1e-4)
        l2_spectrum_keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                                        metrics=['categorical_accuracy'])
        l2_spectrum_keras_model.summary()

        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1,
                                   min_delta=1e-4),
                     ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=4,
                                       verbose=1,
                                       min_lr=1e-10),
                     ModelCheckpoint(filepath="models/{}.h5".format(self.id),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True)]

        history_training = l2_spectrum_keras_model.fit(
            x=generator_first_layer_spectrum(batch_size=batch_size, track_length=self.track_length,
                                             track_time=self.track_time),
            steps_per_epoch=3000,
            epochs=25,
            callbacks=callbacks,
            validation_data=generator_first_layer_spectrum(batch_size=batch_size, track_length=self.track_length,
                                                           track_time=self.track_time),
            validation_steps=300)

        self.keras_model = l2_spectrum_keras_model
        self.convert_history_to_db_format(history_training)

    def evaluate_track_input(self, track):
        assert (track.track_length == self.track_length), "Invalid input track length"

        model_predictions = np.zeros(shape=self.output_categories)
        axis_data_diff = np.zeros(shape=[1, 2, self.track_length, track.n_axes])
        for i in range(track.n_axes):
            axis_data_diff[0, :, :, i] = axis_adaptation_to_net_spectrum(axis_data=track.axes_data[str(i)],
                                                                         track_length=self.track_length)

        for axis in range(track.n_axes):
            input_net = np.zeros(shape=[1, 2, self.track_length, 1])
            input_net[0, :, :, 0] = axis_data_diff[0, :, :, axis]
            model_predictions = (self.keras_model.predict(input_net)[0, :]) + model_predictions
        mean_prediction = np.argmax(model_predictions / track.n_axes)

        return mean_prediction

    def validate_test_data_accuracy(self, n_axes, normalized=True):
        test_batch_size = 100
        ground_truth = np.zeros(shape=test_batch_size)
        predicted_value = np.zeros(shape=test_batch_size)
        for i in range(test_batch_size):
            ground_truth[i] = np.random.choice([0, 1, 2])
            if ground_truth[i] == 0:
                physical_model = FBM.create_random()
            elif ground_truth[i] == 1:
                physical_model = CTRW.create_random()
            else:
                physical_model = TwoStateDiffusion.create_random()

            if ground_truth[i] < 2:
                x_noisy, y_noisy, x, y, t = physical_model.simulate_track(self.track_length, self.track_time)
            else:
                x_noisy, y_noisy, x, y, t, state, switching = physical_model.simulate_track(self.track_length,
                                                                                            self.track_time)

            track = SimulatedTrack(track_length=self.track_length, track_time=self.track_time,
                                   n_axes=n_axes, model_type=physical_model.__class__.__name__)
            track.set_axes_data([x_noisy, y_noisy])
            track.set_time_axis(t)

            predicted_value[i] = self.evaluate_track_input(track=track)

        plot_confusion_matrix_for_layer(layer_name=self.model_name,
                                        ground_truth=ground_truth,
                                        predicted_value=predicted_value,
                                        labels=self.output_categories_labels,
                                        normalized=normalized)

    def output_net_to_labels(self, output_net):
        return self.output_categories_labels[output_net]
