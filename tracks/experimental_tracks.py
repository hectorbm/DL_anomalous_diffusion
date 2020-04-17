from . import tracks
from mongoengine import StringField, ObjectIdField

LABELING_METHODS = ['btx', 'mAb']
EXPERIMENTAL_CONDITIONS = ['Control', 'CDx-Chol', 'CDx']


class ExperimentalTracks(tracks.Tracks):
    labeling_method = StringField(choices=LABELING_METHODS, required=True)
    experimental_condition = StringField(choices=EXPERIMENTAL_CONDITIONS, required=True)
    origin_file = ObjectIdField(required=True)

