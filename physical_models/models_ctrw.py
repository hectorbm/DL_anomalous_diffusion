import numpy as np
from . import models

class CTRW(models.Models):

    min_alpha = 0.1
    max_alpha = 0.9

    def __init__(self, alpha):
        assert (alpha >= self.min_alpha and alpha <= self.max_alpha), "Invalid alpha parameter"
        self.alpha = alpha
        self.beta = 0.5
        self.gamma = 1

    @classmethod
    def create_random(cls):
        random_alpha = np.random.uniform(low=cls.min_alpha, high=cls.max_alpha)
        model = cls(alpha=random_alpha)
        return model

    def mittag_leffler_rand(self, track_length):
        # Generate mittag-leffler random numbers
        t = -np.log(np.random.uniform(size=[track_length,1]))
        u = np.random.uniform(size=[track_length,1])
        w = np.sin(self.beta*np.pi)/np.tan(self.beta*np.pi*u)-np.cos(self.beta*np.pi)
        t = t*((w**1/(self.beta)))
        t = self.gamma*t
        return t

    def symmetric_alpha_levy(self, track_length):
        alpha_levy_dist = 2
        gamma_levy_dist = self.gamma **(self.alpha/2)
        # Generate symmetric alpha-levi random numbers
        u = np.random.uniform(size=[track_length,1])
        v = np.random.uniform(size=[track_length,1])

        phi = np.pi*(v-0.5)
        w = np.sin(alpha_levy_dist*phi)/np.cos(phi)
        z = -1*np.log(u)*np.cos(phi)
        z = z/np.cos((1-alpha_levy_dist)*phi)
        x = gamma_levy_dist*w*z**(1-(1/alpha_levy_dist))

        return x

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def simulate_track(self,track_length,T):

        jumpsX = self.mittag_leffler_rand(track_length)
        rawTimeX = np.cumsum(jumpsX)
        tX = rawTimeX*(T)/np.max(rawTimeX)
        tX = np.reshape(tX,[len(tX),1])

        jumpsY = self.mittag_leffler_rand(track_length)
        rawTimeY = np.cumsum(jumpsY)
        tY = rawTimeY*(T)/np.max(rawTimeY)
        tY = np.reshape(tY,[len(tY),1])

        x = self.symmetric_alpha_levy(track_length)
        x = np.cumsum(x)
        x = np.reshape(x,[len(x),1])

        y = self.symmetric_alpha_levy(track_length)
        y = np.cumsum(y)
        y = np.reshape(y,[len(y),1])

        tOut = np.arange(0,track_length,1)*T/track_length
        xOut = np.zeros([track_length,1])
        yOut = np.zeros([track_length,1])
        for i in range(track_length):
            xOut[i,0] = x[self.find_nearest(tX,tOut[i]),0]
            yOut[i,0] = y[self.find_nearest(tY,tOut[i]),0]
        
        x = xOut[:,0]
        y = yOut[:,0]
        t = tOut

        #Scale to 10.000 nm * 10.000 nm
        if np.min(x) < 0:
            x =  x + np.absolute(np.min(x)) # Add offset to x
        if np.min(y) < 0:
            y = y + np.absolute(np.min(y)) #Add offset to y 
        #Scale to nm and add a random offset
        if np.max(x) != 0:
            x = x * (1/np.max(x)) * np.min([10000,((track_length**1.1)*np.random.uniform(low=3, high=4))])
        else:
            x = x * np.min([10000,((track_length**1.1)*np.random.uniform(low=3, high=4))])
        if np.max(y) != 0:
            y = y * (1/np.max(y)) * np.min([10000,((track_length**1.1)*np.random.uniform(low=3, high=4))])
        else:
            y = y * np.min([10000,((track_length**1.1)*np.random.uniform(low=3, high=4))])

        if np.max(x) < 10000:
            offset_x = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(x)))
            x = x + offset_x 
        if np.max(y) < 10000:
            offset_y = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(y)))
            y = y + offset_y

        x,y = add_noise(x,y,track_length)

        return x,y,t