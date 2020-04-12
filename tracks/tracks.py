import numpy as np
import matplotlib.pyplot as plt


class Tracks:

    def __init__(self, track_length, track_time, n_axes):
        self.track_length = track_length
        self.track_time = track_time
        self.n_axes = n_axes
        self.axes_data = np.zeros(shape=[self.n_axes, self.track_length])
        self.time_axis = np.zeros(shape=self.track_length)

    def plot_xy(self):
        assert (self.n_axes == 2), "Track n_axes ~= 2!"
        plt.xlabel("Position X [nm]")
        plt.ylabel("Position Y [nm]")
        plt.plot(self.axes_data[0], self.axes_data[1])
        plt.show()

    def plot_axis_with_time(self, n_axis):
        assert (self.n_axes >= n_axis > 0), "Invalid axis"
        plt.xlabel("Time")
        plt.ylabel(f"Axis {n_axis - 1}")
        plt.plot(self.time_axis, self.axes_data[n_axis - 1])
        plt.show()

    def plot_axis_velocity(self, n_axis):
        assert (self.n_axes >= n_axis > 0), "Invalid axis"
        plt.xlabel("Time")
        plt.ylabel(f"Velocity Axis {n_axis - 1}")
        dt = self.time_axis[1] - self.time_axis[0]
        velocity_axis = np.diff(self.axes_data[n_axis - 1]) * (1 / dt)
        plt.plot(self.time_axis[:len(self.time_axis) - 1], velocity_axis)
        plt.show()

    def get_track_time(self):
        return self.track_time

    def get_track_length(self):
        return self.track_length

    def set_axes_data(self, axes_data):
        assert (axes_data.shape == self.axes_data.shape)
        self.axes_data = axes_data

    def set_time_axis(self, time_axis_data):
        assert (time_axis_data.shape == self.time_axis.shape)
        self.time_axis = time_axis_data
