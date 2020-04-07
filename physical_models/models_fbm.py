import numpy as np
from scipy import fftpack
from . import models
from . import models_noise

class FBM(models.Models):

    sub_diff_min_max = [0.1,0.42]
    super_diff_min_max = [0.58,0.9]

    def __init__(self, hurst_exp):
        self.hurst_exp = hurst_exp
    
    @classmethod
    def create_random(cls):
        fbm_type = np.random.choice(["subdiff","superdiff","brownian"])
        if fbm_type == "subdiff":
            model = cls.create_random_subdiffusive()
        elif fbm_type == "superdiff":
            model = cls.create_random_superdiffusive()
        else:
            model = cls.create_random_brownian()
        return model

    @classmethod
    def create_random_superdiffusive(cls, hurst_exp=None):
        if hurst_exp is not None:
            assert (hurst_exp >= cls.super_diff_min_max[0] and hurst_exp<=cls.super_diff_min_max[1]), "Invalid Hurst Exponent"
            model = cls(hurst_exp=hurst_exp)
            
        else: 
            random_hurst_exp = np.random.uniform(low=cls.super_diff_min_max[0], high=cls.super_diff_min_max[1])
            model = cls(hurst_exp=random_hurst_exp)
        return model

    @classmethod
    def create_random_subdiffusive(cls, hurst_exp=None):
        if hurst_exp is not None:
            assert (hurst_exp >= cls.sub_diff_min_max[0] and hurst_exp<=cls.sub_diff_min_max[1]), "Invalid Hurst Exponent"
            model = cls(hurst_exp=hurst_exp)
        else: 
            random_hurst_exp = np.random.uniform(low=cls.sub_diff_min_max[0], high=cls.sub_diff_min_max[1])
            model = cls(hurst_exp=random_hurst_exp)
        return model

    @classmethod
    def create_random_brownian(cls, use_exact_exp=False):
        if use_exact_exp:
            model = cls(hurst_exp=0.5)
        else:
            random_brownian_hurst_exp = np.random.uniform(low=cls.sub_diff_min_max[1], high=cls.super_diff_min_max[0])
            model = cls(hurst_exp=random_brownian_hurst_exp)
        return model

    def simulate_track(self, track_length,T):

        r = np.zeros(track_length+1) # first row of circulant matrix
        r[0] = 1
        idx = np.arange(1,track_length+1,1)
        r[idx] = 0.5*((idx+1)**(2*self.hurst_exp) - 2*idx**(2*self.hurst_exp) + (idx-1)**(2*self.hurst_exp))
        r = np.concatenate((r,r[np.arange(len(r)-2,0,-1)]))

        # get eigenvalues through fourier transform
        lamda = np.real(fftpack.fft(r))/(2*track_length)

        # get trajectory using fft: dimensions assumed uncoupled
        x = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*track_length)) + 1j*np.random.normal(size=(2*track_length))))
        x = track_length**(-self.hurst_exp)*np.cumsum(np.real(x[:track_length])) # rescale
        x = ((T**self.hurst_exp)*x)# resulting traj. in x
        
        y = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*track_length)) + 1j*np.random.normal(size=(2*track_length))))
        y = track_length**(-self.hurst_exp)*np.cumsum(np.real(y[:track_length])) # rescale
        y = ((T**self.hurst_exp)*y) # resulting traj. in y

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

        x,y = models_noise.add_noise(x,y,track_length)
        
        if np.min(x) < 0:
            x =  x + np.absolute(np.min(x)) # Add offset to x
        if np.min(y) < 0:
            y = y + np.absolute(np.min(y)) #Add offset to y 
        
        if np.max(x) < 10000:
            offset_x = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(x)))
            x = x + offset_x 
        if np.max(y) < 10000:
            offset_y = np.ones(shape=x.shape) * np.random.uniform(low=0, high=(10000-np.max(y)))
            y = y + offset_y
        
    
        t = np.arange(0,track_length,1)/track_length
        t = t*T # scale for final time T


        return x,y,t

    def get_diffusion_type(self):
        if self.hurst_exp >= self.sub_diff_min_max[0] and self.hurst_exp <=self.sub_diff_min_max[1]:
            return "subdiffusive"
        elif self.hurst_exp > self.sub_diff_min_max[1] and self.hurst_exp < self.super_diff_min_max[0]:
            return "brownian"
        else:
            return "superdiffusive"