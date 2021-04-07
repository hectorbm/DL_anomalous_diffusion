from . import tracks
from mongoengine import StringField, ObjectIdField, FloatField, DictField, ListField, BooleanField
import numpy as np

LABELING_METHODS = ['BTX', 'mAb']
EXPERIMENTAL_CONDITIONS = ['Control', 'CDx-Chol', 'CDx']
L2_output_categories_labels = ["Subdiffusive", "Brownian", "Superdiffusive"]
L1_output_categories_labels = ["fBm", "CTRW", "2-State-OD"]


class ExperimentalTracks(tracks.Tracks):
    labeling_method = StringField(choices=LABELING_METHODS, required=True)
    experimental_condition = StringField(choices=EXPERIMENTAL_CONDITIONS, required=True)
    origin_file = ObjectIdField(required=False)
    immobile = BooleanField(required=False)


    # Output Nets
    l1_classified_as = StringField(choices=L1_output_categories_labels, required=False)
    l1_error = FloatField(min_value=0,max_value=1,required=False)
    
    l2_classified_as = StringField(choices=L2_output_categories_labels, required=False)
    l2_error = FloatField(min_value=0,max_value=1,required=False)

    diffusion_coefficient_brownian = FloatField(required=False)
    diffusion_coefficient_brownian_error = FloatField(min_value=0,max_value=1,required=False)
    
    hurst_exponent_fbm = FloatField(required=False, min_value=0, max_value=1)
    hurst_mae = FloatField(min_value=0,max_value=1,required=False)

    track_states = ListField(required=False)
    segments = ListField(required=False)
    transitions = DictField(required=False)

    def set_l1_classified(self, label):
        self.l1_classified_as = label

    def set_l2_classified(self, label):
        self.l2_classified_as = label

    def set_hurst_exponent(self, exp_val):
        self.hurst_exponent_fbm = exp_val

    def set_track_states(self, states):
        self.track_states = states

    def set_seq_diffusion_coefficient(self, pos, d_coefficient):
        self.seq_diffusion_coefficient[pos] = d_coefficient

    # Only for two state segmentation
    def compute_segments(self):
        self.segments = []
        segment_state = self.track_states[0]
        step = 0

        # For initial segment
        segment_initial_step = step
        segment_final_step = -1
        
        for current_state in self.track_states:
            if current_state != segment_state:
                # End segment    
                segment_final_step = step - 1
                segment = self.create_segment(segment_state, segment_initial_step, segment_final_step)
                self.add_segment(segment)
                # Start a new segment
                segment_state = current_state
                segment_initial_step = step
                segment_final_step = -1
            
            if step == self.track_length - 1:
                # The last step ends the last segment
                segment_final_step = self.track_length - 1
                segment = self.create_segment(segment_state, segment_initial_step, segment_final_step)
                self.add_segment(segment)
            
            step += 1

        
    def create_segment(self, state, initial_step, final_step):
        segment = {'state':state,
                   'initial_step': initial_step,
                   'final_step': final_step,
                   'length': final_step - initial_step + 1,
                   'residence_time': self.time_axis[final_step]-self.time_axis[initial_step]}
        return segment

    def compute_confinement_region(self, segment):
        x_segment, y_segment = self.get_segment_axes(segment)
        distance_x = max(x_segment) - min(x_segment)
        distance_y = max(y_segment) - min(y_segment)
        area = distance_x * distance_y
        segment['confinement_area'] = area


    def add_segment(self, segment):
        if segment['initial_step'] < segment['final_step']:
            self.segments.append(segment)


    def get_brownian_state_segments(self):
        return [segment for segment in self.segments if segment['state'] == 0 and segment['length']>3]


    def get_od_state_segments(self):
        return [segment for segment in self.segments if segment['state'] == 1 and segment['length']>4]

    def get_segment_axes(self, segment):
        x_segment = self.axes_data['0'][segment['initial_step']: segment['final_step']+1]
        y_segment = self.axes_data['1'][segment['initial_step']: segment['final_step']+1]
        return x_segment, y_segment

    def get_segment_time_axis(self, segment):
        return self.time_axis[segment['initial_step']: segment['final_step']+1]

    def create_track_from_segment(self, segment):
        x_segment, y_segment = self.get_segment_axes(segment)
        axes_data = np.zeros(shape=(2,segment['length']))
        axes_data[0] = x_segment
        axes_data[1] = y_segment
        time_segment = np.zeros(shape=(1,segment['length']))
        time_segment[0] = self.get_segment_time_axis(segment)
        sub_track = ExperimentalTracks(track_length=segment['length'],
                                       track_time=segment['residence_time'],
                                       n_axes=2,
                                       labeling_method=self.labeling_method,
                                       experimental_condition=self.experimental_condition)
        sub_track.set_time_axis(time_segment[0])
        sub_track.set_axes_data(axes_data)
        
        return sub_track

    def compute_transitions(self):
        if len(self.segments) > 0:
            self.transitions = {'OD_to_Brownian':0, 
                                'Brownian_to_OD':0}
            
            last_segment_state = self.segments[0]['state']
            last_segment_length = self.segments[0]['length']
            last_segment_step = self.segments[0]['final_step']
            
            for segment in self.segments:
                if segment['state'] != last_segment_state and segment['initial_step'] == last_segment_step + 1:
                    if segment['state'] == 0 and segment['length'] > 3 and last_segment_length > 4:
                        self.transitions['OD_to_Brownian'] += 1
                    elif segment['state'] == 1 and segment['length'] > 4 and last_segment_length > 3:
                        self.transitions['Brownian_to_OD'] += 1
                
                last_segment_state = segment['state']
                last_segment_step = segment['final_step']
                last_segment_length = segment['length']
    
    def compute_two_state_segments_data(self):
        # Detect all segments
        self.compute_segments()
        # Compute confinement area for segments in OD state
        od_state_segments = self.get_od_state_segments()
        for segment in od_state_segments:
            self.compute_confinement_region(segment)
        # Analyze transitions between states
        self.compute_transitions()