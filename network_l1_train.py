from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv1D
from keras.layers import Input,GlobalMaxPooling1D
from keras.layers import Dropout, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from generator import generator_first_layer

def train(batchsize, steps,T,sigma):
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
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc','mse'])
    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss',
                        patience=20,
                        verbose=0,
                        min_delta=1e-4),
            ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            patience=4,
                            verbose=0,
                            min_lr=1e-9)]

    model.fit_generator(
            generator=generator_first_layer(batchsize=batchsize,track_length=steps,track_time=T,sigma=sigma),
            steps_per_epoch=4000,
            epochs=50,
            callbacks=callbacks,
            validation_data=generator_first_layer(batchsize=batchsize,track_length=steps,track_time=T,sigma=sigma),
            validation_steps=10)
    return model



if __name__ == "__main__":
    train(batchsize=64,steps=100,T=1.2,sigma=0)    