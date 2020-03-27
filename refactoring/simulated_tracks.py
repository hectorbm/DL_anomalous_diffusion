import matplotlib.pyplot as plt
import numpy as np

class SimulatedTrack:

    def __init__(self, track_length, time_length, model, n_axes):
        self.track_length = track_length
        self.track_time = time_length
        self.model = model
        self.n_axes = n_axes
        self.axes_data = np.zeros(shape=[self.n_axes,self.track_length])
        self.time_axis = np.zeros(shape=self.track_length)

    def plot_xy(self):
        assert (self.n_axes == 2), "Track n_axes ~= 2!"
        plt.xlabel("Position X")
        plt.ylabel("Position Y")
        plt.plot(self.axes_data[0], self.axes_data[1])
        plt.show()

    def set_axes_data(self, axes_data):
        assert (axes_data.shape == self.axes_data.shape)
        self.axes_data = axes_data

    def set_time_axis(self, time_axis_data):
        assert (time_axis_data.shape == self.time_axis.shape)
        self.time_axis = time_axis_data