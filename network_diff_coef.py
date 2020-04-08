from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv1D
from keras.layers import Input,GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from generators import generator_coeff_network

def train_diff_network(batchsize, track_length, track_time, sigma, model_id,state):

    initializer = 'he_normal'
    f = 32
    inputs = Input((2,1))

    #
    x2 = Conv1D(f,2,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = GlobalMaxPooling1D()(x2)

    dense = Dense(512,activation='relu')(x2)
    dense = Dense(256,activation='relu')(dense)
    dense2 = Dense(1,activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=dense2)

    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss= 'mse',metrics=['mse'])
    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss',
                        patience=20,
                        verbose=1,
                        min_delta=1e-4),
            ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            patience=4,
                            verbose=1,
                            min_lr=1e-12),
            ModelCheckpoint(filepath="models/{}.h5".format(model_id),
                            monitor='val_loss',
                            save_best_only=True,
                            mode='min',
                            save_weights_only=False)]


    model.fit_generator(generator=generator_coeff_network(batchsize,track_length,track_time,sigma,state),
            steps_per_epoch=500,
            epochs=100,
            callbacks=callbacks,
            validation_data=generator_coeff_network(batchsize,track_length,track_time,sigma,state),
            validation_steps=50)

    return model

if __name__ == "__main__":
    model = train_diff_network(batchsize=64,track_length=120,track_time=2,sigma=0,model_id="net_diff_coeff_1",state=0)
