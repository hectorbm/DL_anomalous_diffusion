import numpy as np
from . import tracks


class ExperimentalTracks(tracks.Tracks):

    def __init__(self, track_length, track_time, n_axes, noise_level, label):
        assert (label == 'btx' or label == 'mAb')
        super().__init__(track_length, track_time, n_axes)
        self.label = label

    def set_predicted_model_type(self, model_type_prediction, network_id):
        # Network id: unique id assigned to each network trained
        self.predicted_model_type = {
            'model_type_prediction': model_type_prediction,
            'network_id': network_id}
