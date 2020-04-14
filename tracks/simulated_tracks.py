from . import tracks


class SimulatedTrack(tracks.Tracks):

    def __init__(self, track_length, track_time, n_axes, model_type, origin_track_id):
        super().__init__(track_length, track_time, n_axes)

        self.model_type = model_type
        self.origin_track_id = origin_track_id


    def get_model_type(self):
        return self.model_type
