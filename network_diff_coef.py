from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv1D,Dropout
from keras.layers import Input,GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from generators import generator_coeff_network
from tools.load_model import load_model_from_file
from physical_models.models_two_state_diffusion import TwoStateDiffusion
import numpy as np
import matplotlib.pyplot as plt

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

    optimizer = Adam(lr=1e-4)
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
            ModelCheckpoint(filepath="{}.h5".format(model_id),
                            monitor='val_loss',
                            save_best_only=True,
                            mode='min',
                            save_weights_only=False)]


    history = model.fit_generator(generator=generator_coeff_network(batchsize,track_length,track_time,sigma,state),
            steps_per_epoch=1000,
            epochs=100,
            callbacks=callbacks,
            validation_data=generator_coeff_network(batchsize,track_length,track_time,sigma,state),
            validation_steps=100)

    return model,history



if __name__ == "__main__":
    model = train_diff_network(batchsize=64,track_length=30,track_time=0.5,sigma=0,model_id="net_diff_coeff_1_state1",state=1)
    """model = load_model_from_file("models/net_diff_coeff_1_state1.h5")
    for i in range(10):
        out = np.zeros([1,2,1])
        label = np.zeros([1,1])
        two_state_model = TwoStateDiffusion.create_random()
        x,y,t = two_state_model.simulate_track_only_state0(track_length=30,T=0.5,noise=True)

        dx = np.diff(x,axis=0)
        dy = np.diff(y,axis=0)
        mx = np.mean(np.abs(dx),axis=0)
        my = np.mean(np.abs(dy),axis=0)
        sx = np.std(dx,axis=0)
        sy = np.std(dy,axis=0)

        out[0,:,0] = [mx,sx]
        pred_x = model.predict(out[:,:,:])
        
        out[0,:,0] = [my,sy]
        pred_y = model.predict(out[:,:,:])
        
        pred = (pred_x + pred_y) / 2 
        pred = pred * 0.049 + 0.001
        label = two_state_model.get_D_state0()
        plt.scatter(i,label, color='b')
        plt.scatter(i,pred[0,0], color='r')
        print('Ground truth:{:.4}  prediction:{:.4}  error:{:.4}'.format(label,pred[0,0],np.abs(pred-label)[0,0]))
    plt.show()"""


