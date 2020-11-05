from . import tracks
from mongoengine import StringField


class SimulatedTrack(tracks.Tracks):
    model_type = StringField(required=True)
