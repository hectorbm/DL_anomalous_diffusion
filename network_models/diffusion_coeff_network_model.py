from . import network_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from network_models.generators import generator_diffusion_coefficient_network


class DiffusionCoefficientNetworkModel(network_model.NetworkModel):

    def train_network(self, batch_size, track_time, diffusion_model_state, noise_reduction_model):
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

        optimizer = Adam(lr=1e-4)
        diffusion_coefficient_keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
        diffusion_coefficient_keras_model.summary()

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

        history_training = diffusion_coefficient_keras_model.fit(
            x=generator_diffusion_coefficient_network(batch_size,
                                                      self.track_length,
                                                      track_time,
                                                      diffusion_model_state,
                                                      noise_reduction_model),
            steps_per_epoch=1000,
            epochs=100,
            callbacks=callbacks,
            validation_data=
            generator_diffusion_coefficient_network(batch_size,
                                                    self.track_length,
                                                    track_time,
                                                    diffusion_model_state,
                                                    noise_reduction_model),
            validation_steps=100)

        self.history = history_training.history
        self.keras_model = diffusion_coefficient_keras_model

    def evaluate_track_input(self, track):
        pass