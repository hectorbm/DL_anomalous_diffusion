import math

import numpy as np
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalAveragePooling1D, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from Networks.generators import generator_state_net, generator_state_net_validation, generate_batch_states_net
from PhysicalModels.two_states_obstructed_diffusion import TwoStateObstructedDiffusion
from Tools.analysis_tools import plot_confusion_matrix_for_layer
from Tracks.simulated_tracks import SimulatedTrack
from . import network

# TODO: Justify MSE loss function


def convert_output_to_db(states_net_output):
    return states_net_output.tolist()


class StateDetectionNetworkModel(network.NetworkModel):
    output_categories_labels = ["State-0", "State-1"]
    model_name = 'States Detection Network'

    net_params = {
        'lr': 0.001,
        'training_set_size': 50000,
        'validation_set_size': 12500,
        'batch_size': 16,
        'amsgrad': False,
        'epsilon': 1e-6
    }
    # For analysis of hyper-params
    analysis_params = {
        'lr': [1e-2, 1e-3, 1e-4, 1e-5],
        'amsgrad': [False, True],
        'batch_size': [8, 16, 32],
        'epsilon': [1e-6, 1e-7, 1e-8]
    }

    def train_network(self, batch_size):
        # Generate training set
        x_data, y_data = generate_batch_states_net(self.net_params['training_set_size'],
                                                   self.track_length,
                                                   self.track_time)

        state_detection_keras_model = self.build_model()
        state_detection_keras_model.summary()

        callbacks = [
            EarlyStopping(monitor="val_loss",
                          min_delta=1e-3,
                          patience=5,
                          verbose=1,
                          mode="min")
        ]
        if self.hiperparams_opt:
            validation_generator = generator_state_net_validation(batch_size=batch_size,
                                                                  track_length=self.track_length,
                                                                  track_time=self.track_time,
                                                                  validation_set_size=self.net_params['validation_set_size'])
        else:
            validation_generator = generator_state_net(batch_size=batch_size,
                                                       track_length=self.track_length,
                                                       track_time=self.track_time)

        history_training = state_detection_keras_model.fit(
            x=x_data,
            y=y_data,
            epochs=50,
            batch_size=self.net_params['batch_size'],
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=math.floor(self.net_params['validation_set_size'] / self.net_params['batch_size']),
            shuffle=True)

        self.convert_history_to_db_format(history_training)
        self.keras_model = state_detection_keras_model
        self.keras_model.save(filepath="Models/{}".format(self.id))

        if self.hiperparams_opt:
            self.params_training = self.net_params

    def build_model(self):
        # Networks filters and kernels
        initializer = 'he_normal'
        filters_size = 32
        x1_kernel_size = 4
        x2_kernel_size = 2
        x3_kernel_size = 3
        x4_kernel_size = 10
        x5_kernel_size = 20
        inputs = Input(shape=(self.track_length, 1))
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
        x1 = GlobalAveragePooling1D()(x1)
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
        x2 = GlobalAveragePooling1D()(x2)
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
        x3 = GlobalAveragePooling1D()(x3)
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
        x4 = GlobalAveragePooling1D()(x4)
        x5 = Conv1D(filters=filters_size, kernel_size=x5_kernel_size, padding='same', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x5 = BatchNormalization()(x5)
        x5 = GlobalAveragePooling1D()(x5)
        x_concat = concatenate(inputs=[x1, x2, x3, x4, x5])
        dense_1 = Dense(units=(self.track_length * 2), activation='relu')(x_concat)
        dense_2 = Dense(units=self.track_length, activation='relu')(dense_1)
        output_network = Dense(units=self.track_length, activation='sigmoid')(dense_2)

        state_detection_keras_model = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=self.net_params['lr'],
                         epsilon=self.net_params['epsilon'],
                         amsgrad=self.net_params['amsgrad'])

        state_detection_keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

        return state_detection_keras_model

    def evaluate_track_input(self, track):
        assert track.track_length == self.track_length, "Invalid track length"

        if self.keras_model is None:
            self.load_model_from_file()

        model_predictions = np.zeros(shape=self.track_length)

        for axis in range(track.n_axes):
            input_net = np.zeros(shape=[1, self.track_length, 1])
            input_net[0, :, 0] = track.axes_data[str(axis)] - np.mean(track.axes_data[str(axis)])
            model_predictions = (self.keras_model.predict(input_net)[0, :]) + model_predictions

        mean_prediction = model_predictions / track.n_axes

        # Convert to state values
        for i in range(self.track_length):
            if mean_prediction[i] < 0.5:
                mean_prediction[i] = 0
            else:
                mean_prediction[i] = 1

        return mean_prediction

    def validate_test_data_accuracy(self, n_axes, normalized=True):
        test_batch_size = 100
        ground_truth = np.zeros(shape=(test_batch_size, self.track_length))
        predicted_value = np.zeros(shape=(test_batch_size, self.track_length))
        for i in range(test_batch_size):
            physical_model = TwoStateObstructedDiffusion.create_random()

            switching = False
            while not switching:
                x_noisy, y_noisy, x, y, t, state, switching = physical_model.simulate_track(self.track_length,
                                                                                            self.track_time)

            ground_truth[i, :] = state
            track = SimulatedTrack(track_length=self.track_length, track_time=self.track_time,
                                   n_axes=n_axes, model_type=physical_model.__class__.__name__)
            track.set_axes_data([x_noisy, y_noisy])
            track.set_time_axis(t)

            predicted_value[i, :] = self.evaluate_track_input(track=track)
        ground_truth = ground_truth.flatten()
        predicted_value = predicted_value.flatten()
        plot_confusion_matrix_for_layer(layer_name=self.model_name,
                                        ground_truth=ground_truth,
                                        predicted_value=predicted_value,
                                        labels=self.output_categories_labels,
                                        normalized=normalized)
