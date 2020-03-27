import numpy as np
from scipy import fftpack

class FBM:

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
    def create_random_subdiffusive(cls, hurst_exp=None):
        if hurst_exp is not None:
            assert (hurst_exp >= 0.58 and hurst_exp<=0.9), "Invalid Hurst Exponent"
            model = cls(hurst_exp=hurst_exp)
            
        else: 
            random_hurst_exp = np.random.uniform(low=0.58, high=0.9)
            model = cls(hurst_exp=random_hurst_exp)
        return model

    @classmethod
    def create_random_superdiffusive(cls, hurst_exp=None):
        if hurst_exp is not None:
            assert (hurst_exp >= 0.1 and hurst_exp<=0.42), "Invalid Hurst Exponent"
            model = cls(hurst_exp=hurst_exp)
        else: 
            random_hurst_exp = np.random.uniform(low=0.1, high=0.42)
            model = cls(hurst_exp=random_hurst_exp)
        return model

    @classmethod
    def create_random_brownian(cls, use_exact_exp=False):
        if use_exact_exp:
            model = cls(hurst_exp=0.5)
        else:
            random_brownian_hurst_exp = np.random.uniform(low=0.42, high=0.58)
            model = cls(hurst_exp=random_brownian_hurst_exp)
        return model

    def simulate_track(self, track_length=1000,T=15):

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

        t = np.arange(0,track_length+1,1)/track_length
        t = t*T # scale for final time T


        return x,y,t
