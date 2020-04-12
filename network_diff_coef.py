from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from generators import generator_diffusion_coefficient_network


def train_diff_network(batch_size, track_length, track_time, model_id, diffusion_model_state, noise_reduction_model):
    initializer = 'he_normal'
    filters_size = 32
    x_kernel_size = 2

    inputs = Input(shape=(2, 1))

    x = Conv1D(filters=filters_size, kernel_size=x_kernel_size, padding='same', activation='relu',
               kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling1D()(x)

    dense_1 = Dense(units=512, activation='relu')(x)
    dense_2 = Dense(units=256, activation='relu')(dense_1)
    output_network = Dense(units=1, activation='sigmoid')(dense_2)
    diffusion_coefficient_network_model = Model(inputs=inputs, outputs=output_network)

    optimizer = Adam(lr=1e-4)
    diffusion_coefficient_network_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    diffusion_coefficient_network_model.summary()

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

    history = diffusion_coefficient_network_model.fit(x=generator_diffusion_coefficient_network(batch_size,
                                                                                                track_length,
                                                                                                track_time,
                                                                                                diffusion_model_state,
                                                                                                noise_reduction_model),
                                                      steps_per_epoch=1000,
                                                      epochs=100,
                                                      callbacks=callbacks,
                                                      validation_data=
                                                      generator_diffusion_coefficient_network(batch_size,
                                                                                              track_length,
                                                                                              track_time,
                                                                                              diffusion_model_state,
                                                                                              noise_reduction_model),
                                                      validation_steps=100)

    return diffusion_coefficient_network_model, history


if __name__ == "__main__":
    model = train_diff_network(batch_size=64, track_length=30, track_time=0.5,
                               model_id="net_diff_coeff_1_state1", diffusion_model_state=0, noise_reduction_model=None)
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
