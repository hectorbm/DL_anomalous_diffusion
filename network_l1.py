"""
    Currently all the N.N were developed using Keras, in the future a migration to Pytorch should be 
    a better option.
    Currently a 2 step classifier outperforms a single neural network because of the similarities between 
    a 2-state diffusion model and a fbm/brownian model, in particular when D1 and D2(diff coef) are really 
    similar or a very intense switching behavior, which leads to a mean diffusivity D* 
    Two state model was added using Monte Carlo simulations introduced in:
    'Time-averaged mean square displacement for switching diffusion' by Denis Grebenkov 
"""
from keras.models import Model, load_model
from keras.layers import Dense,BatchNormalization,Conv1D
from keras.layers import Input,GlobalMaxPooling1D
from keras.layers import Dropout, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from generator import generator_first_layer,axis_adaptation_to_net,generate_batch_of_samples
from tools.analysis_tools import plot_confusion_matrix_for_layer
import datetime
import numpy as np

def train_l1_net(batchsize, steps, T, sigma, model_id):
    initializer = 'he_normal'
    f = 32

    inputs = Input((steps-1,1))

    x1 = Conv1D(f,4,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f,4,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f,4,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalMaxPooling1D()(x1)


    x2 = Conv1D(f,2,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f,2,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f,2,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalMaxPooling1D()(x2)


    x3 = Conv1D(f,3,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f,3,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f,3,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = GlobalMaxPooling1D()(x3)


    x4 = Conv1D(f,10,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f,10,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f,10,dilation_rate=8,padding='causal',activation='relu',kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = GlobalMaxPooling1D()(x4)


    x5 = Conv1D(f,20,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x5 = BatchNormalization()(x5)
    x5 = GlobalMaxPooling1D()(x5)


    con = concatenate([x1,x2,x3,x4,x5])
    dense = Dense(512,activation='relu')(con)
    dense = Dense(128,activation='relu')(dense)
    dense2 = Dense(3,activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=dense2)

    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['categorical_accuracy'])
    model.summary()

   

    callbacks = [EarlyStopping(monitor='val_loss',
                        patience=20,
                        verbose=0,
                        min_delta=1e-4),
            ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            patience=4,
                            verbose=0,
                            min_lr=1e-9),
            ModelCheckpoint("models/{}.h5".format(model_id),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True)]

    model.fit(
            x=generator_first_layer(batchsize=batchsize,track_length=steps,track_time=T,sigma=sigma),
            steps_per_epoch=4000,
            epochs=50,
            callbacks=callbacks,
            validation_data=generator_first_layer(batchsize=batchsize,track_length=steps,track_time=T,sigma=sigma),
            validation_steps=10)
    return model

def load_model_from_file(filename):
    try:
        model = load_model(filename,compile=True)
    except ValueError:
        print("File doesn`t exist!")
    return model

def evaluate_model_multi_axis(model,axis_data_diff,n_axes,track_length,time_length):
    model_predictions = np.zeros(shape=3)
    for axis in range(n_axes):
        input_net = np.zeros([1,track_length-1,1])
        input_net[0,:,0] = axis_data_diff[:,axis]
        model_predictions = (model.predict(input_net)[0,:]) + model_predictions
    mean_prediction = np.argmax(model_predictions / n_axes)
    return mean_prediction

def validate_test_data_over_model(model,n_axes,track_length,time_length,sigma):
    test_batchsize = 1000
    axis_data_diff, ground_truth = generate_batch_of_samples(test_batchsize,track_length,time_length,sigma)
    ground_truth = np.reshape(ground_truth,test_batchsize)
    predictions = np.zeros(shape=test_batchsize)
    print("Please wait, evaluating test data ...")
    for sample in range(test_batchsize):
        predictions[sample] = evaluate_model_multi_axis(model,axis_data_diff[sample],n_axes,track_length,time_length)
    plot_confusion_matrix_for_layer(model,1,ground_truth,predictions,["fBm","CTRW","Two-State"],True)
    


if __name__ == "__main__":
    #For testing
    #train_l1_net(batchsize=64,steps=100,T=1.2,sigma=0,model_id='first_layer_1')    
    model = load_model_from_file("models/first_layer_1.h5")
    validate_test_data_over_model(model,2,100,1.2,0)