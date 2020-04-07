import numpy as np

def add_noise(x,y,track_length):
    #New error formula
    mean_error = 40
    sigma_error = 15
    error_x = np.random.normal(loc=mean_error/2, scale=sigma_error/2, size=track_length)
    error_y = np.random.normal(loc=mean_error/2, scale=sigma_error/2, size=track_length)

    for i in range(track_length):
            #Add noise to x coordinate
        if np.random.choice(["left","right"]) == "left":
            x[i] = x[i] - error_x[i]
        else:
            x[i] = x[i] + error_x[i]
            #Add noise to y coordinate
        if np.random.choice(["left","right"]) == "left":
            y[i] = y[i] - error_y[i]
        else: 
            y[i] = y[i] + error_y[i]
    return x,y