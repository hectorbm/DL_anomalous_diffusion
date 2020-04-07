import numpy as np
from . import models
from . import models_noise

class TwoStateDiffusion:
    """
    State-0: Free Diffusion
    State-1: Confined Diffusion
    """
    # De aca se puede sacar algunas conclusiones 
    #http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-sim-BM.pdf
    # Saque el 2 del scaling de grebenkov, cual es la explicacion de 
    # utilizar esta constante que sale de la nada? shetchman en su 
    #simulacion de brownian no lo utiliza tampoco.
    def __init__(self, k_state0, k_state1, D_state0, D_state1):
        self.k_state0 = k_state0
        self.k_state1 = k_state1
        self.D_state0 = D_state0 * 1000000 # Convert from um^2 -> nm^2
        self.D_state1 = D_state1 * 1000000
        self.beta0 = 1
        self.beta1 = 2

    @classmethod
    def create_random(cls):
        # k_state(i) dimensions = 1 / frame
        # D_state(i) dimensions = um^2 * s^(-beta)
        D_state0 = np.random.uniform(low=0.05 ,high=0.1) 
        D_state1 = np.random.uniform(low=0.001 , high=0.05)
        k_state0 = np.random.uniform(low=0.01 ,high=0.08) 
        k_state1 = np.random.uniform(low=0.007 ,high=0.2)
        model = cls(k_state0, k_state1, D_state0, D_state1)
        return model
    @classmethod
    def create_with_coefficients(cls,k_state0, k_state1, D_state0, D_state1):
        assert(D_state0 >= 0.05 and D_state0 <= 0.3), "Invalid Diffusion coeficient state-0"
        assert(D_state1 >= 0.001 and D_state1 <= 0.05), "Invalid Diffusion coeficient state-1"
        assert(k_state0 >= 0.01 and k_state0 <= 0.08), "Invalid switching rate state-0"
        assert(k_state0 >= 0.007 and k_state0 <= 0.2), "Invalid switching rate state-1"
        return cls(k_state0, k_state1, D_state0, D_state1)

    def get_D_state0(self):
        return self.D_state0 / 1000000
    def get_D_state1(self):
        return self.D_state1 / 1000000

    def simulate_track(self, track_length, T,noise=True):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        #Residence time
        res_time0 = 1 / self.k_state0
        res_time1 = 1 / self.k_state1

        #Compute each t_state acording to exponential laws
        t_state0 =  np.random.exponential(scale=res_time0, size=track_length) 
        t_state1 =  np.random.exponential(scale=res_time1, size=track_length)

        #Set initial t_state for each state
        t_state0_next = 0
        t_state1_next = 0

        #Pick an initial state from a random choice
        current_state = np.random.choice([0, 1])

        #Detect real switching behavior
        switching = ((current_state == 0) and (int(np.ceil(t_state0[t_state0_next])) < track_length)) or ((current_state == 1) and (int(np.ceil(t_state1[t_state1_next])) < track_length))

        #Fill state array
        state = np.zeros(shape=track_length)
        i = 0

        while i < track_length:
            if current_state == 1:
                current_state_length = int(np.ceil(t_state1[t_state1_next]))

                if (current_state_length + i) < track_length:
                    state[i:(i + current_state_length)] = np.ones(shape=current_state_length)
                else:
                    state[i:track_length] = np.ones(shape=(track_length-i))

                current_state = 0 #Set state from 1->0
            else:
                current_state_length = int(np.ceil(t_state0[t_state0_next]))
                current_state = 1 #Set state from 0->1

            i += current_state_length

        for i in range(len(state)):
            if state[i] == 0:
                x[i] = x[i] * np.sqrt(self.D_state0 * ((T/track_length) ** self.beta0))
                y[i] = y[i] * np.sqrt(self.D_state0 * ((T/track_length) ** self.beta0))
            else:
                x[i] = x[i] * np.sqrt(self.D_state1 * ((T/track_length) ** self.beta1))
                y[i] = y[i] * np.sqrt(self.D_state1 * ((T/track_length) ** self.beta1))
        x = np.cumsum(x)
        y = np.cumsum(y)

        # Add noise
        if noise:
            x,y = models_noise.add_noise(x,y,track_length)

        if np.min(x) < 0:
            x =  x + np.absolute(np.min(x)) # Add offset to x
        if np.min(y) < 0:
            y = y + np.absolute(np.min(y)) #Add offset to y

        offset_x = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(x)))
        offset_y = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(y)))

        x = x + offset_x 
        y = y + offset_y

        t = np.arange(0,track_length,1)/track_length
        t = t*T

        return x,y,t,state,switching

    def simulate_track_only_state0(self, track_length, T,noise=True):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        for i in range(track_length):
            x[i] = x[i] * np.sqrt(self.D_state0 * ((T/track_length) ** self.beta0))
            y[i] = y[i] * np.sqrt(self.D_state0 * ((T/track_length) ** self.beta0))

        x = np.cumsum(x)
        y = np.cumsum(y)

        # Add noise
        if noise:
            x,y = models_noise.add_noise(x,y,track_length)

        if np.min(x) < 0:
            x =  x + np.absolute(np.min(x)) # Add offset to x
        if np.min(y) < 0:
            y = y + np.absolute(np.min(y)) #Add offset to y

        offset_x = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(x)))
        offset_y = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(y)))

        x = x + offset_x 
        y = y + offset_y

        t = np.arange(0,track_length,1)/track_length
        t = t*T

        return x,y,t
    def simulate_track_only_state1(self, track_length, T,noise=True):
        x = np.random.normal(loc=0, scale=1, size=track_length)
        y = np.random.normal(loc=0, scale=1, size=track_length)

        for i in range(track_length):
            x[i] = x[i] * np.sqrt(self.D_state1 * ((T/track_length) ** self.beta1))
            y[i] = y[i] * np.sqrt(self.D_state1 * ((T/track_length) ** self.beta1))

        x = np.cumsum(x)
        y = np.cumsum(y)

        # Add noise
        if noise:
            x,y = models_noise.add_noise(x,y,track_length)

        if np.min(x) < 0:
            x =  x + np.absolute(np.min(x)) # Add offset to x
        if np.min(y) < 0:
            y = y + np.absolute(np.min(y)) #Add offset to y

        offset_x = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(x)))
        offset_y = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(y)))

        x = x + offset_x 
        y = y + offset_y

        t = np.arange(0,track_length,1)/track_length
        t = t*T

        return x,y,t
