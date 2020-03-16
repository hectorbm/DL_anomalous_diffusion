from keras.utils import to_categorical
from models import CTRW, two_state_switching_diffusion,fbm_diffusion
import numpy as np

def generate_samples(batchsize=32, steps=5000,T=15, steps_actual=1000, noise=0.1):
    while True:
        #Select randomly a type of motion
        out = np.zeros([batchsize,steps_actual-1,1])
        label = np.zeros([batchsize,1])

        for i in range(batchsize):
            simulation_type = np.random.choice(["fbm","CTRW","Brownian","2-State-Diff"])

            if simulation_type == "fbm":
                fbm_type = np.random.choice(["super-diff", "sub-diff"])
                label[i,0] = 0
                if fbm_type == "super-diff":
                    H = np.random.uniform(low=0.1, high=0.42)
                else: 
                    H = np.random.uniform(low=0.58, high=0.9)
                x,y,t = fbm_diffusion(steps, H, T)

            elif simulation_type == "CTRW":
                label[i,0] = 1
                alpha_sample = np.random.uniform(low=0.05,high=0.9)
                x,y,t = CTRW(n=steps, alpha=alpha_sample, gamma=1,T=T)

            elif simulation_type == "Brownian":
                label[i,0] = 2
                H = np.random.uniform(low=0.43, high=0.57)
                x,y,t = fbm_diffusion(steps, H, T)

            else: #2-State-Diffusion case
                """ State0 Normal Diffusion 
                    State1 Confined Diffusion """
                label[i,0] = 3
                k_state0 = np.random.uniform(low=0.01 ,high=0.09)
                k_state1 = np.random.uniform(low=0.02 ,high=0.07)
                D_state0 = np.random.uniform(low=0.01 ,high=0.08) 
                D_state1 = np.random.uniform(low=0.001 , high=0.01)
                x,y,t,state = two_state_switching_diffusion(steps, k_state0, k_state1, D_state0, D_state1, T)

            #Add noise to the data
            x_noise,y_noise = generate_gaussian_noise(x,y,steps_actual,noise)

            # Only 1 dimension is being used because the simulations are homogeneous
            x1 = np.reshape(x,[1,len(x)])
            x1 = x1-np.mean(x1)
            x_n = x1[0,:steps_actual] + x_noise
            dx = np.diff(x_n)
            
            out[i,:,0] = dx
    
        label = to_categorical(label,num_classes=4)
        yield out, label


def generate_gaussian_noise(x, y,steps,sigma):
    noise_x = sigma * np.std(np.diff(x,axis=0),axis=0) * np.random.randn(1,steps)
    noise_y = sigma * np.std(np.diff(y,axis=0),axis=0) * np.random.randn(1,steps)
    return noise_x, noise_y

