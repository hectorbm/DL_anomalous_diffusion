from keras.models import Model, load_model
from keras.layers import Dense,BatchNormalization,Conv1D
from keras.layers import Input,GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dropout, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from generators import generator_state_net,axis_adaptation_to_net,generate_batch_of_samples_state_net
from tools.analysis_tools import plot_confusion_matrix_for_layer
import datetime
import numpy as np

def train_states_net(batchsize, steps, T, sigma, model_id):
    initializer = 'he_normal'
    f = 32

    inputs = Input((steps,1))

    x1 = Conv1D(f,4,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f,4,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f,4,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling1D()(x1)


    x2 = Conv1D(f,2,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f,2,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f,2,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling1D()(x2)


    x3 = Conv1D(f,3,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f,3,dilation_rate=2,padding='causal',activation='relu',kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f,3,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = GlobalAveragePooling1D()(x3)


    x4 = Conv1D(f,10,padding='causal',activation='relu',kernel_initializer=initializer)(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f,10,dilation_rate=4,padding='causal',activation='relu',kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f,10,dilation_rate=8,padding='causal',activation='relu',kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = GlobalAveragePooling1D()(x4)

    x5 = Conv1D(f,20,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x5 = BatchNormalization()(x5)
    x5 = GlobalAveragePooling1D()(x5)


    con = concatenate([x1,x2,x3,x4,x5])
    dense = Dense((steps*2),activation='relu')(con)
    dense = Dense(steps,activation='relu')(con)
    dense2 = Dense(steps,activation='sigmoid')(dense)

    model = Model(inputs=inputs, outputs=dense2)

    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer,loss='mse',metrics=['mse'])
    model.summary()


    callbacks = [EarlyStopping(monitor='val_mse',
                        patience=20,
                        verbose=0,
                        min_delta=1e-4),
            ReduceLROnPlateau(monitor='val_mse',
                            factor=0.1,
                            patience=4,
                            verbose=0,
                            min_lr=1e-9),
            ModelCheckpoint("models/{}.h5".format(model_id),
                        monitor='val_mse',
                        verbose=1,
                        save_best_only=True)]

    model.fit(
            x=generator_state_net(batchsize=batchsize,track_length=steps,track_time=T,sigma=sigma),
            steps_per_epoch=8000,
            epochs=50,
            callbacks=callbacks,
            validation_data=generator_state_net(batchsize=batchsize,track_length=steps,track_time=T,sigma=sigma),
            validation_steps=10)
    return model



def load_model_from_file(filename):
    try:
        model = load_model(filename,compile=True)
    except ValueError:
        print("File doesn`t exist!")
    return model

def evaluate_model_multi_axis(model,axis_data,n_axes,track_length,time_length):
    model_predictions = np.zeros(shape=track_length)
    for axis in range(n_axes):
        input_net = np.zeros([1,track_length,1])
        input_net[0,:,0] = axis_data[:,axis]
        model_predictions = (model.predict(input_net)[0,:]) + model_predictions
    mean_prediction = model_predictions / n_axes
    for i in range(track_length):
        if mean_prediction[i]<0.5:
            mean_prediction[i] = 0
        else:
            mean_prediction[i] = 1
    return mean_prediction

def validate_test_data_over_model(model,n_axes,track_length,time_length,sigma):
    test_batchsize = 1
    axis_data, ground_truth = generate_batch_of_samples_state_net(test_batchsize,track_length,time_length,sigma)
    ground_truth = np.reshape(ground_truth,(test_batchsize,track_length))
    predictions = np.zeros(shape=(test_batchsize,track_length))
    print("Please wait, evaluating test data ...")
    for sample in range(test_batchsize):
        predictions[sample,:] = evaluate_model_multi_axis(model,axis_data[sample],n_axes,track_length,time_length)
    
    plot_confusion_matrix_for_layer(model,'State-Detection',ground_truth.flatten(),predictions.flatten(),["State-0","State-1"],True)
    

if __name__ == "__main__":
    #For testing
    #train_states_net(batchsize=32,steps=100,T=1.2,sigma=0,model_id='state_net_1')    
    model = load_model_from_file("models/state_net_1.h5")
    validate_test_data_over_model(model,2,100,1.2,0)
    