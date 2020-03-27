import numpy as np

class CTRW:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @classmethod
    def random_initialization(cls):
        random_alpha = np.random.uniform(low=0.1, high=0.9)
        model = CTRW(alpha=random_alpha, beta=0.5, gamma=1)
        return model

    def mittag_leffler_rand(self, n = 1000):
        # Generate mittag-leffler random numbers
        t = -np.log(np.random.uniform(size=[n,1]))
        u = np.random.uniform(size=[n,1])
        w = np.sin(self.beta*np.pi)/np.tan(self.beta*np.pi*u)-np.cos(self.beta*np.pi)
        t = t*((w**1/(self.beta)))
        t = self.gamma*t
        return t

    def symmetric_alpha_levy(self, n=1000):
        alpha_levy_dist = 2
        gamma_levy_dist = self.gamma **(self.alpha/2)
        # Generate symmetric alpha-levi random numbers
        u = np.random.uniform(size=[n,1])
        v = np.random.uniform(size=[n,1])

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

        x = self.symmetric_alpha_levy(n=track_length)
        x = np.cumsum(x)
        x = np.reshape(x,[len(x),1])

        y = self.symmetric_alpha_levy(n=track_length)
        y = np.cumsum(y)
        y = np.reshape(y,[len(y),1])

        tOut = np.arange(0,track_length,1)*T/track_length
        xOut = np.zeros([track_length,1])
        yOut = np.zeros([track_length,1])
        for i in range(track_length):
            xOut[i,0] = x[self.find_nearest(tX,tOut[i]),0]
            yOut[i,0] = y[self.find_nearest(tY,tOut[i]),0]

        return xOut,yOut,tOut
