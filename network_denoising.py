from keras.models import Model, load_model
from keras.layers import Dense,BatchNormalization,Conv1D
from keras.layers import Input,GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dropout, concatenate,Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
from generators import generator_denoising_net

def train_denoising_net(batchsize, steps, T, sigma, model_id):
    initializer = 'he_normal'
    f = 32

    inputs = Input((steps,1))
    x1 = Dense(steps,activation='relu')(inputs)
    x1 = Conv1D(filters=20,kernel_size=2,padding='causal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)
    x1 = Dense(steps,activation='relu')(x1)

    dense = Dense(steps)(x1)

    model = Model(inputs=inputs, outputs=dense)

    optimizer = Adam(lr=1e-3)
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
            ModelCheckpoint("{}.h5".format(model_id),
                        monitor='val_mse',
                        verbose=1,
                        save_best_only=True)]

    model.fit(
            x=generator_denoising_net(batchsize=batchsize,track_length=steps,track_time=T,sigma=sigma,state=1),
            steps_per_epoch=1000,
            epochs=50,
            callbacks=callbacks,
            validation_data=generator_denoising_net(batchsize=batchsize,track_length=steps,track_time=T,sigma=sigma,state=1),
            validation_steps=100)
    return model