from keras.models import Model
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from network_models.generators import generator_first_layer
from . import network_model


class L1NetworkModel(network_model.NetworkModel):

    def train_network(self, batch_size, track_time):
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
        output_network = Dense(units=3, activation='softmax')(dense_2)
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
                                                                      track_time=track_time),
                                              steps_per_epoch=4000,
                                              epochs=50,
                                              callbacks=callbacks,
                                              validation_data=generator_first_layer(batch_size=batch_size,
                                                                                    track_length=self.track_length,
                                                                                    track_time=track_time),
                                              validation_steps=100)
        self.history = history_training.history
        self.keras_model = l1_keras_model

    def evaluate_track_input(self, track):
        pass
