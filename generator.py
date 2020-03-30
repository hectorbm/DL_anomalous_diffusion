import numpy as np
from keras.utils import to_categorical
from models_ctrw import CTRW
from models_fbm import FBM
from models_two_state_diffusion import TwoStateDiffusion


def generator_first_layer(batchsize,track_length,track_time,sigma):
    while True:
        out = np.zeros([batchsize,track_length-1,1])
        label = np.zeros([batchsize,1])
        T_sample = np.random.choice(np.arange(track_time,track_time+1,0.5))
        steps_sample = int(np.random.choice(np.arange(track_length, np.ceil(track_length*1.05),1)))

        for i in range(batchsize):
            model_sample = np.random.choice(["fbm","ctrw","two-state"])
            if model_sample == "fbm":
                model = FBM.create_random()
                x,y,t = model.simulate_track(steps_sample,T_sample)
                label[i,0] = 0
            elif model_sample == "ctrw":
                model = CTRW.create_random()
                x,y,t = model.simulate_track(steps_sample,T_sample)
                label[i,0] = 1
            else: 
                model = TwoStateDiffusion.create_random()
                switching = False
                while not switching:
                    x,y,t,state,switching = model.simulate_track(steps_sample,T_sample)
                label[i,0] = 2

            x1 = np.reshape(x,[1,len(x)])
            x1 = x1 - np.mean(x1)
            x_n = x1[0,:track_length] 
            dx = np.diff(x_n)
            out[i,:,0] = dx

        label = to_categorical(label,num_classes=3)
        yield out, label

