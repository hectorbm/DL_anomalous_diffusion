from mongoengine import StringField

from networks.generators import generator_hurst_exp_network
from networks.network_model import NetworkModel
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam


class HurstExponentNetworkModel(NetworkModel):
    fbm_type = StringField(choices=["subdiffusive", "superdiffusive"], required=True)
    model_name = "Hurst Exponent Network"

    def train_network(self, batch_size):
        inputs = Input(shape=(2, self.track_length))
        x = LSTM(units=64, return_sequences=True, input_shape=(2, self.track_length))(inputs)
        x = LSTM(units=16)(x)
        x = Dense(units=128, activation='selu')(x)
        output_network = Dense(units=1, activation='sigmoid')(x)

        hurst_exp_keras_model = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=1e-4)
        hurst_exp_keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        hurst_exp_keras_model.summary()

        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   verbose=1,
                                   min_delta=1e-4),
                     ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=4,
                                       verbose=1,
                                       min_lr=1e-12),
                     ModelCheckpoint(filepath="test_h.h5",
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True)]

        history_training = hurst_exp_keras_model.fit(
            x=generator_hurst_exp_network(batch_size=batch_size, track_length=self.track_length,
                                          track_time=self.track_time, fbm_type=self.fbm_type),
            steps_per_epoch=500,
            epochs=35,
            callbacks=callbacks,
            validation_data=generator_hurst_exp_network(batch_size=batch_size, track_length=self.track_length,
                                                        track_time=self.track_time, fbm_type=self.fbm_type),
            validation_steps=50)

        self.convert_history_to_db_format(history_training)
        self.keras_model = hurst_exp_keras_model

    def evaluate_track_input(self, track):
        pass

    def validate_test_data_mse(self, n_axes):
        pass
