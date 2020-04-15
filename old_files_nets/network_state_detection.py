from keras.models import Model
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalAveragePooling1D, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from network_models.generators import generator_state_net, generate_batch_of_samples_state_net
from tools.analysis_tools import plot_confusion_matrix_for_layer
import numpy as np

from tools.load_model import load_model_from_file


def train_network(batch_size, track_length, track_time, model_id):
    initializer = 'he_normal'
    filters_size = 32
    x1_kernel_size = 4
    x2_kernel_size = 2
    x3_kernel_size = 3
    x4_kernel_size = 10
    x5_kernel_size = 20

    inputs = Input(shape=(track_length, 1))

    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=2, padding='causal', activation='relu',
                kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=4, padding='causal', activation='relu',
                kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling1D()(x1)

    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=2, padding='causal', activation='relu',
                kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=4, padding='causal', activation='relu',
                kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling1D()(x2)

    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=2, padding='causal', activation='relu',
                kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=4, padding='causal', activation='relu',
                kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = GlobalAveragePooling1D()(x3)

    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=4, padding='causal', activation='relu',
                kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=8, padding='causal', activation='relu',
                kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = GlobalAveragePooling1D()(x4)

    x5 = Conv1D(filters=filters_size, kernel_size=x5_kernel_size, padding='same', activation='relu',
                kernel_initializer=initializer)(inputs)
    x5 = BatchNormalization()(x5)
    x5 = GlobalAveragePooling1D()(x5)

    x_concat = concatenate(inputs=[x1, x2, x3, x4, x5])
    dense_1 = Dense(units=(track_length * 2), activation='relu')(x_concat)
    dense_2 = Dense(units=track_length, activation='relu')(dense_1)
    output_network = Dense(units=track_length, activation='sigmoid')(dense_2)

    state_detection_network_model = Model(inputs=inputs, outputs=output_network)

    optimizer = Adam(lr=1e-4)
    state_detection_network_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    state_detection_network_model.summary()

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

    history = state_detection_network_model.fit(
        x=generator_state_net(batch_size=batch_size, track_length=track_length, track_time=track_time),
        steps_per_epoch=8000,
        epochs=50,
        callbacks=callbacks,
        validation_data=generator_state_net(batch_size=batch_size, track_length=track_length, track_time=track_time),
        validation_steps=200)
    return state_detection_network_model, history


def evaluate_model_multi_axis(model, axis_data, n_axes, track_length):
    model_predictions = np.zeros(shape=track_length)
    for axis in range(n_axes):
        input_net = np.zeros(shape=[1, track_length, 1])
        input_net[0, :, 0] = axis_data[:, axis]
        model_predictions = (model.predict(input_net)[0, :]) + model_predictions

    mean_prediction = model_predictions / n_axes

    # Convert to state values
    for i in range(track_length):
        if mean_prediction[i] < 0.5:
            mean_prediction[i] = 0
        else:
            mean_prediction[i] = 1
    return mean_prediction


def validate_test_data_over_model(model, n_axes, track_length, track_time):
    test_batch_size = 10
    axis_data, ground_truth = generate_batch_of_samples_state_net(batch_size=test_batch_size,
                                                                  track_length=track_length,
                                                                  track_time=track_time)

    ground_truth = np.reshape(ground_truth, newshape=(test_batch_size, track_length))
    predictions = np.zeros(shape=(test_batch_size, track_length))

    for sample in range(test_batch_size):
        predictions[sample, :] = evaluate_model_multi_axis(model=model,
                                                           axis_data=axis_data[sample],
                                                           n_axes=n_axes,
                                                           track_length=track_length)

    plot_confusion_matrix_for_layer(layer_name='State-Detection',
                                    ground_truth=ground_truth.flatten(),
                                    predicted_value=predictions.flatten(),
                                    labels=["State-0", "State-1"],
                                    normalized=True)


if __name__ == "__main__":
    # For testing
    #train_states_net(batch_size=32, track_length=100, track_time=1.2, model_id='state_net_1')
    model = load_model_from_file("../models/state_net_1.h5")
    validate_test_data_over_model(model=model, n_axes=2, track_length=100, track_time=1.2)
