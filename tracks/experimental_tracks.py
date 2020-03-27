import numpy as np
class ExperimentalTracks:

    def __init__(self, track_length, time_length, n_axes, noise_level):
        self.track_length = track_length
        self.track_time = time_length
        self.n_axes = n_axes
        self.predicted_model = None
        self.axes_data = np.zeros(shape=[self.n_axes,self.track_length])
        self.time_axis = np.zeros(shape=self.track_length)
        self.noise_level = noise_level

