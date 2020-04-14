import numpy as np
import matplotlib.pyplot as plt
from mongoengine import Document, IntField, ListField


class Tracks(Document):
    track_length = IntField(min_value=1, required=True)
    track_time = IntField(min_value=0, required=True)
    n_axes = IntField(min_value=1, required=True)
    time_axis = ListField(required=True)
    x_axis_data = ListField(required=True)
    y_axis_data = ListField(required=True)

    meta = {'allow_inheritance': True}

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
        x_axis = axes_data[0].tolist()
        y_axis = axes_data[1].tolist()
        assert len(x_axis) == self.track_length
        assert len(y_axis) == self.track_length
        self.x_axis_data = x_axis
        self.y_axis_data = y_axis
        self.axes_data = axes_data

    def get_axes_data(self):
        return self.axes_data

    def set_time_axis(self, time_axis_data):
        time_axis_data = time_axis_data.tolist()
        assert (len(time_axis_data) == self.track_length)
        self.time_axis = time_axis_data
