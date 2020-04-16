from keras.models import Model
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from networks.generators import generator_first_layer, axis_adaptation_to_net
import numpy as np

from physical_models.models_ctrw import CTRW
from physical_models.models_fbm import FBM
from physical_models.models_two_state_diffusion import TwoStateDiffusion
from tools.analysis_tools import plot_confusion_matrix_for_layer
from tracks.simulated_tracks import SimulatedTrack
from . import network_model


class L1NetworkModel(network_model.NetworkModel):
    output_categories = 3
    output_categories_labels = ["fBm", "CTRW", "Two-State"]
    model_name = 'L1 Network'

    def train_network(self, batch_size):
        initializer = 'he_normal'
        filters_size = 32
        x1_kernel_size = 4
        x2_kernel_size = 2
        x3_kernel_size = 3
        x4_kernel_size = 10
        x5_kernel_size = 20

        inputs = Input(shape=(self.track_length - 1, 1))

        x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=2, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)
        x1 = GlobalMaxPooling1D()(x1)

        x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=2, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x2)
        x2 = BatchNormalization()(x2)
        x2 = GlobalMaxPooling1D()(x2)

        x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=2, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x3)
        x3 = BatchNormalization()(x3)
        x3 = GlobalMaxPooling1D()(x3)

        x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=8, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x4)
        x4 = BatchNormalization()(x4)
        x4 = GlobalMaxPooling1D()(x4)

        x5 = Conv1D(filters=filters_size, kernel_size=x5_kernel_size, padding='same', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x5 = BatchNormalization()(x5)
        x5 = GlobalMaxPooling1D()(x5)

        x_concat = concatenate(inputs=[x1, x2, x3, x4, x5])
        dense_1 = Dense(units=512, activation='relu')(x_concat)
        dense_2 = Dense(units=128, activation='relu')(dense_1)
        output_network = Dense(units=self.output_categories, activation='softmax')(dense_2)
        l1_keras_model = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=1e-5)
        l1_keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        l1_keras_model.summary()

        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=10,
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

        history_training = l1_keras_model.fit(x=generator_first_layer(batch_size=batch_size,
                                                                      track_length=self.track_length,
                                                                      track_time=self.track_time),
                                              steps_per_epoch=4000,
                                              epochs=50,
                                              callbacks=callbacks,
                                              validation_data=generator_first_layer(batch_size=batch_size,
                                                                                    track_length=self.track_length,
                                                                                    track_time=self.track_time),
                                              validation_steps=100)

        self.convert_history_to_db_format(history_training)
        self.keras_model = l1_keras_model

    def evaluate_track_input(self, track):
        assert (track.track_length == self.track_length), "Invalid input track length"

        if self.keras_model is None:
            self.load_model_from_file()

        model_predictions = np.zeros(shape=self.output_categories)

        axis_data_diff = np.zeros(shape=[1, self.track_length - 1, track.n_axes])
        for i in range(track.n_axes):
            axis_data_diff[0, :, i] = axis_adaptation_to_net(axis_data=track.axes_data[str(i)],
                                                             track_length=self.track_length)

        for axis in range(track.n_axes):
            input_net = np.zeros([1, self.track_length - 1, 1])
            input_net[0, :, 0] = axis_data_diff[0, :, axis]
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
                x_noisy, y_noisy, x, y, t, state, switching = physical_model.simulate_track(self.track_length, self.track_time)

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
