from . import tracks


class SimulatedTrack(tracks.Tracks):

    def __init__(self, track_length, time_length, n_axes, model_type, noise_level):
        super().__init__(track_length, time_length, n_axes)
        self.noise_level = noise_level
        self.model_type = model_type
        # Add noise data to all axes
        # Compute noise and data separately to allow noise analysis!

    def get_model_type(self):
        return self.model_type
