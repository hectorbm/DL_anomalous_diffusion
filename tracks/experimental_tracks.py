from . import tracks
from mongoengine import StringField, ObjectIdField, FloatField, DictField, ListField

LABELING_METHODS = ['btx', 'mAb']
EXPERIMENTAL_CONDITIONS = ['Control', 'CDx-Chol', 'CDx']
L2_output_categories_labels = ["Subdiffusive", "Brownian", "Superdiffusive"]
L1_output_categories_labels = ["fBm", "CTRW", "2-State-OD"]
FILE_fps = 50


class ExperimentalTracks(tracks.Tracks):
    labeling_method = StringField(choices=LABELING_METHODS, required=True)
    experimental_condition = StringField(choices=EXPERIMENTAL_CONDITIONS, required=True)
    origin_file = ObjectIdField(required=True)
    # Output Nets
    l1_classified_as = StringField(choices=L1_output_categories_labels, required=False)
    l2_classified_as = StringField(choices=L2_output_categories_labels, required=False)

    diffusion_coefficient_brownian = FloatField(required=False)
    hurst_exponent_fbm = FloatField(required=False, min_value=0, max_value=1)

    track_states = ListField(required=False)
    axes_data_noise_reduced = DictField(required=False)

    frames = ListField(required=True)
    seq_initial_frame = ListField(required=False)
    seq_final_frame = ListField(required=False)
    seq_res_time = ListField(required=False)
    confinement_regions_area = ListField(required=False)
    seq_diffusion_coefficient = ListField(required=False)

    def set_l1_classified(self, label):
        self.l1_classified_as = label

    def set_l2_classified(self, label):
        self.l2_classified_as = label

    def set_hurst_exponent(self, exp_val):
        self.hurst_exponent_fbm = exp_val

    def set_d_coefficient(self, d_value):
        self.diffusion_coefficient_brownian = d_value

    def set_track_states(self, states):
        self.track_states = states

    def set_frames(self, frames):
        self.frames = frames

    def compute_sequences_res_time(self):
        self.seq_res_time = []
        for i in range(len(self.seq_initial_frame)):
            self.seq_res_time.append((self.frames[self.seq_final_frame[i]] - self.frames[self.seq_initial_frame[i]]) * (1 / FILE_fps))

    def compute_sequences_length(self):
        # Compute sequences initial and final frame, instantiate conf regions ans diff coef
        assert self.l1_classified_as == "2-State-OD" and self.track_length > 0
        self.seq_initial_frame = []
        self.seq_final_frame = []
        self.seq_res_time = []
        self.seq_initial_frame.append(0)
        current_state = self.track_states[0]

        for i in range(1, self.track_length):
            if not (current_state == self.track_states[i]):
                self.seq_final_frame.append(i - 1)
                current_state = self.track_states[i]
                self.seq_initial_frame.append(i)

        if len(self.seq_initial_frame) == len(self.seq_final_frame) + 1:
            self.seq_final_frame.append(self.track_length - 1)

        self.confinement_regions_area = [0 for i in range(len(self.seq_initial_frame))]
        self.seq_diffusion_coefficient = [0 for i in range(len(self.seq_initial_frame))]

    def compute_confinement_regions(self):
        assert len(self.seq_initial_frame) == len(self.seq_final_frame)

        for i in range(len(self.seq_initial_frame)):
            if self.track_states[self.seq_initial_frame[i]] == 1:
                x = self.axes_data[str(0)][self.seq_initial_frame[i]: self.seq_final_frame[i] + 1]
                y = self.axes_data[str(1)][self.seq_initial_frame[i]: self.seq_final_frame[i] + 1]
                dist_x = max(x) - min(x)
                dist_y = max(y) - min(y)
                if dist_x > 0 and dist_y > 0:
                    self.confinement_regions_area[i] = dist_x * dist_y

    def set_seq_diffusion_coefficient(self, pos, d_coefficient):
        self.seq_diffusion_coefficient[pos] = d_coefficient

    def get_res_time_state(self, state):
        assert state == 0 or state == 1
        state_res_time = []
        for i in range(len(self.seq_res_time)):
            if self.track_states[self.seq_initial_frame[i]] == state:
                state_res_time.append(self.seq_res_time[i])

        return state_res_time
