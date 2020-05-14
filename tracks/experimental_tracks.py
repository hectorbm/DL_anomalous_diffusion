from networks.l1_network_model import L1NetworkModel
from networks.l2_network_model import L2NetworkModel
from . import tracks
from mongoengine import StringField, ObjectIdField, FloatField, DictField, ListField

LABELING_METHODS = ['btx', 'mAb']
EXPERIMENTAL_CONDITIONS = ['Control', 'CDx-Chol', 'CDx']


class ExperimentalTracks(tracks.Tracks):
    labeling_method = StringField(choices=LABELING_METHODS, required=True)
    experimental_condition = StringField(choices=EXPERIMENTAL_CONDITIONS, required=True)
    origin_file = ObjectIdField(required=True)
    # Output Nets
    l1_classified_as = StringField(choices=L1NetworkModel.output_categories_labels, required=False)
    l2_classified_as = StringField(choices=L2NetworkModel.output_categories_labels, required=False)

    diffusion_coefficient_brownian = FloatField(required=False)
    hurst_exponent_fbm = FloatField(required=False, min_value=0, max_value=1)

    track_states = ListField(required=False)
    axes_data_noise_reduced = DictField(required=False)

    def set_l1_classified(self, label):
        self.l1_classified_as = label

    def set_l2_classified(self, label):
        self.l1_classified_as = label

    def set_hurst_exponent(self, exp_val):
        self.hurst_exponent_fbm = exp_val

    def track_states(self, states):
        self.track_states = states
