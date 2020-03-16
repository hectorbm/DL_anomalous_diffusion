import numpy as np
from scipy import stats,fftpack
import matplotlib.pyplot as plt

#fBm simulation

def fbm_diffusion(n=1000,H=1,T=15):

    # first row of circulant matrix
    r = np.zeros(n+1)
    r[0] = 1
    idx = np.arange(1,n+1,1)
    r[idx] = 0.5*((idx+1)**(2*H) - 2*idx**(2*H) + (idx-1)**(2*H))
    r = np.concatenate((r,r[np.arange(len(r)-2,0,-1)]))
    
    # get eigenvalues through fourier transform
    lamda = np.real(fftpack.fft(r))/(2*n)
    
    # get trajectory using fft: dimensions assumed uncoupled
    x = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*n)) + 1j*np.random.normal(size=(2*n))))
    x = n**(-H)*np.cumsum(np.real(x[:n])) # rescale
    x = ((T**H)*x)# resulting traj. in x
    y = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*n)) + 1j*np.random.normal(size=(2*n))))
    y = n**(-H)*np.cumsum(np.real(y[:n])) # rescale
    y = ((T**H)*y) # resulting traj. in y

    t = np.arange(0,n,1)/n
    t = t*T # scale for final time T
    

    return x,y,t

#CTRW simulation
# Generate mittag-leffler random numbers
#Used for random time jumps
def mittag_leffler_rand(beta = 0.5, n = 1000, gamma = 1):
    t = -np.log(np.random.uniform(size=[n,1]))
    u = np.random.uniform(size=[n,1])
    w = np.sin(beta*np.pi)/np.tan(beta*np.pi*u)-np.cos(beta*np.pi)
    t = t*((w**1/(beta)))
    t = gamma*t
    
    return t

# Generate symmetric alpha-levi random numbers
# Used for random jumps
def symmetric_alpha_levy(alpha = 0.5,n=1000,gamma = 1):
    u = np.random.uniform(size=[n,1])
    v = np.random.uniform(size=[n,1])
    
    phi = np.pi*(v-0.5)
    w = np.sin(alpha*phi)/np.cos(phi)
    z = -1*np.log(u)*np.cos(phi)
    z = z/np.cos((1-alpha)*phi)
    x = gamma*w*z**(1-(1/alpha))
    
    return x

# needed for CTRW
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def CTRW(n=1000,alpha=1,gamma=1,T=40):
    jumpsX = mittag_leffler_rand(alpha,n,gamma)

    rawTimeX = np.cumsum(jumpsX)
    tX = rawTimeX*(T)/np.max(rawTimeX)
    tX = np.reshape(tX,[len(tX),1])
    
    jumpsY = mittag_leffler_rand(alpha,n,gamma)
    rawTimeY = np.cumsum(jumpsY)
    tY = rawTimeY*(T)/np.max(rawTimeY)
    tY = np.reshape(tY,[len(tY),1])
    
    x = symmetric_alpha_levy(alpha=2,n=n,gamma=gamma**(alpha/2))
    x = np.cumsum(x)
    x = np.reshape(x,[len(x),1])
    
    y = symmetric_alpha_levy(alpha=2,n=n,gamma=gamma**(alpha/2))
    y = np.cumsum(y)
    y = np.reshape(y,[len(y),1])
    
    tOut = np.arange(0,n,1)*T/n
    xOut = np.zeros([n,1])
    yOut = np.zeros([n,1])
    for i in range(n):
        xOut[i,0] = x[find_nearest(tX,tOut[i]),0]
        yOut[i,0] = y[find_nearest(tY,tOut[i]),0]
    
    return xOut,yOut,tOut

def two_state_switching_diffusion(n, k_state0, k_state1, D_state0, D_state1, T):
    x = np.random.normal(loc=0, scale=1, size=n)
    y = np.random.normal(loc=0, scale=1, size=n)
    
    #Residence time
    res_time0 = 1 / k_state0
    res_time1 = 1 / k_state1

    #Compute each t_state acording to exponential laws 
    t_state0 = np.random.exponential(scale=res_time0, size=n)
    t_state1 = np.random.exponential(scale=res_time1, size=n)

    #Set initial t_state for each state
    t_state0_next = 0
    t_state1_next = 0

    #Pick an initial state from a random choice
    current_state = np.random.choice([0, 1])
    
    #Fill state array 
    state = np.zeros(shape=n)
    i = 0
    while i < n:
        if current_state == 1:
            current_state_length = int(np.ceil(t_state1[t_state1_next]))
            
            if (current_state_length + i) < n:
                state[i:(i + current_state_length)] = np.ones(shape=current_state_length)
            else:
                state[i:n] = np.ones(shape=(n-i))
            
            current_state = 0 #Set state from 1->0
        else:
            current_state_length = int(np.ceil(t_state0[t_state0_next]))
            current_state = 1 #Set state from 0->1

        i += current_state_length

    for i in range(len(state)):
        if state[i] == 0:
            x[i] = x[i] * np.sqrt(2 * D_state0)
            y[i] = y[i] * np.sqrt(2 * D_state0)
        else:
            x[i] = x[i] * np.sqrt(2 * D_state1)
            y[i] = y[i] * np.sqrt(2 * D_state1)
            
    x = np.cumsum(x)
    y = np.cumsum(y)
    t = np.arange(0,n,1)/n
    t = t*T 
    return x,y,t,state