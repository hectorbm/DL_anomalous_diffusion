"""
    Currently all the N.N were developed using Keras, in the future a migration to Pytorch should be 
    a better option.
    Currently a 2 step classifier outperforms a single neural network because of the similarities between 
    a 2-state diffusion model and a fbm/brownian model, in particular when D1 and D2(diff coefficient) are really
    similar or a very intense switching behavior, which leads to a mean diffusivity D = (D1+D2) / 2
    Two state model was added using Monte Carlo simulations introduced in:
    'Time-averaged mean square displacement for switching diffusion' by Denis Grebenkov 
"""
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from network_models.generators import generator_first_layer, generate_batch_of_samples_l1
from tools.analysis_tools import plot_confusion_matrix_for_layer
from tools.load_model import load_model_from_file
import numpy as np


def train_network(batch_size, track_length, track_time, model_id):
    initializer = 'he_normal'
    filters_size = 32
    x1_kernel_size = 4
    x2_kernel_size = 2
    x3_kernel_size = 3
    x4_kernel_size = 10
    x5_kernel_size = 20

    inputs = Input(shape=(track_length - 1, 1))

    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=2, padding='causal', activation='relu',
                kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=4, padding='causal', activation='relu',
                kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalMaxPooling1D()(x1)

    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=2, padding='causal', activation='relu',
                kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=4, padding='causal', activation='relu',
                kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalMaxPooling1D()(x2)

    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=2, padding='causal', activation='relu',
                kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=4, padding='causal', activation='relu',
                kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = GlobalMaxPooling1D()(x3)

    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=4, padding='causal', activation='relu',
                kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=8, padding='causal', activation='relu',
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
    l1_network_model = Model(inputs=inputs, outputs=output_network)

    optimizer = Adam(lr=1e-5)
    l1_network_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    l1_network_model.summary()

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=10,
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

    history = l1_network_model.fit(
        x=generator_first_layer(batch_size=batch_size, track_length=track_length, track_time=track_time),
        steps_per_epoch=4000,
        epochs=50,
        callbacks=callbacks,
        validation_data=generator_first_layer(batch_size=batch_size, track_length=track_length, track_time=track_time),
        validation_steps=100)
    return l1_network_model, history


def evaluate_model_multi_axis(l1_net_model, axis_data_diff, n_axes, track_length):
    model_predictions = np.zeros(shape=3)
    for axis in range(n_axes):
        input_net = np.zeros([1, track_length - 1, 1])
        input_net[0, :, 0] = axis_data_diff[:, axis]
        model_predictions = (l1_net_model.predict(input_net)[0, :]) + model_predictions
    mean_prediction = np.argmax(model_predictions / n_axes)
    return mean_prediction


def validate_with_test_data_over_model(l1_net_model, n_axes, track_length, track_time):
    test_batch_size = 100
    axis_data_diff, ground_truth = generate_batch_of_samples_l1(batch_size=test_batch_size, track_length=track_length,
                                                                track_time=track_time)
    ground_truth = np.reshape(ground_truth, newshape=test_batch_size)
    predictions = np.zeros(shape=test_batch_size)
    print("Please wait, evaluating test data ...")
    for sample in range(test_batch_size):
        predictions[sample] = evaluate_model_multi_axis(l1_net_model=l1_net_model,
                                                        axis_data_diff=axis_data_diff[sample],
                                                        n_axes=n_axes,
                                                        track_length=track_length)
    plot_confusion_matrix_for_layer(layer_name='layer 1',
                                    ground_truth=ground_truth,
                                    predicted_value=predictions,
                                    labels=["fBm", "CTRW", "Two-State"],
                                    normalized=True)


if __name__ == "__main__":
    # For testing
    #train_l1_net(batch_size=64, track_length=100, track_time=1.2, model_id='first_layer_1')
    model = load_model_from_file("../models/first_layer_1.h5")
    validate_with_test_data_over_model(model, n_axes=2, track_length=100, track_time=1.2)
