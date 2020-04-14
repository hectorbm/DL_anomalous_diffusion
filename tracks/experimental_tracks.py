from . import tracks
from mongoengine import StringField, ObjectIdField

LABELING_METHODS = ['btx', 'mAb']
EXPERIMENTAL_CONDITIONS = ['control', 'CDx-Chol', 'CDx']


class ExperimentalTracks(tracks.Tracks):
    labeling_method = StringField(choices=LABELING_METHODS, required=True)
    experimental_condition = StringField(choices=EXPERIMENTAL_CONDITIONS, required=True)
    origin_file = ObjectIdField(required=True)


    def set_predicted_model_type(self, model_type_prediction, network_id):
        # Network id: unique id assigned to each network trained
        self.predicted_model_type = {
            'model_type_prediction': model_type_prediction,
            'network_id': network_id}
