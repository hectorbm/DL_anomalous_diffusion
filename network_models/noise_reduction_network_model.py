from . import network_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Conv1D, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from network_models.generators import generator_noise_reduction_net


class NoiseReductionNetworkModel(network_model.NetworkModel):

    def train_network(self, batch_size, track_time, diffusion_model_state):
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
                                            track_time=track_time,
                                            diffusion_model_state=diffusion_model_state),
            steps_per_epoch=1000,
            epochs=50,
            callbacks=callbacks,
            validation_data=generator_noise_reduction_net(batch_size=batch_size,
                                                          track_length=self.track_length,
                                                          track_time=track_time,
                                                          diffusion_model_state=diffusion_model_state),
            validation_steps=100)

        self.keras_model = noise_reduction_keras_model
        self.history = history_training.history

    def evaluate_track_input(self, track):
        pass