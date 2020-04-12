from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Conv1D, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from generators import generator_noise_reduction_net


def train_noise_reduction_net(batch_size, track_length, track_time, model_id, diffusion_model_state):
    initializer = 'he_normal'
    filters_size = 20
    kernel_size = 2

    inputs = Input((track_length, 1))
    x = Conv1D(filters=filters_size, kernel_size=kernel_size, padding='causal', activation='relu',
               kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    dense_1 = Dense(units=track_length, activation='relu')(x)
    output_network = Dense(units=track_length)(dense_1)

    noise_reduction_network_model = Model(inputs=inputs, outputs=output_network)

    optimizer = Adam(lr=1e-3)
    noise_reduction_network_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    noise_reduction_network_model.summary()

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=20,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   min_lr=1e-9),
                 ModelCheckpoint(filepath="models/{}.h5".format(model_id),
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)]

    history = noise_reduction_network_model.fit(
            x=generator_noise_reduction_net(batch_size=batch_size,
                                            track_length=track_length,
                                            track_time=track_time,
                                            diffusion_model_state=diffusion_model_state),
            steps_per_epoch=1000,
            epochs=50,
            callbacks=callbacks,
            validation_data=generator_noise_reduction_net(batch_size=batch_size,
                                                          track_length=track_length,
                                                          track_time=track_time,
                                                          diffusion_model_state=diffusion_model_state),
            validation_steps=100)

    return noise_reduction_network_model, history
