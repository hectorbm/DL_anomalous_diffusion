from mongoengine import Document, StringField, FileField
import pandas as pd
import numpy as np
from . import experimental_tracks


def create_tracks_dict(data):
    data_dict = dict()
    for index, row in data.iterrows():
        particle_id = row['track_id']
        x = float(row['x'])
        y = float(row['y'])
        frame = int(row['frame'])
        if particle_id in data_dict:
            data_dict[particle_id]["x"].append(x)
            data_dict[particle_id]["y"].append(y)
            data_dict[particle_id]['frame'].append(frame)
        else:
            data_dict[particle_id] = {"x": [x], "y": [y], 'frame': [frame]}

    data_dict.pop(0)  # Remove non assigned particles

    return data_dict


class File(Document):
    time_length = StringField(required=True)
    experimental_condition = StringField(choices=experimental_tracks.EXPERIMENTAL_CONDITIONS, required=True)
    labeling_method = StringField(choices=experimental_tracks.LABELING_METHODS, required=True)
    raw_file = FileField(required=True)
    file_fps = 58.333333
    file_labeling_method = 'btx'
    file_id_db = 'test_id_value'

    def add_raw_file(self, filename):
        with open(filename, 'rb') as fd:
            self.raw_file.put(fd)

    def load_local_file(self, filename):
        data = pd.read_csv(filename)
        tracks_dict = create_tracks_dict(data)
        tracks_list = self.create_tracks_list(tracks_dict)

        return tracks_list

    def create_tracks_list(self, tracks_dict):
        tracks_list = []
        for key, value in tracks_dict.items():
            x = np.asarray(tracks_dict[key]["x"])
            y = np.asarray(tracks_dict[key]["y"])
            frames = tracks_dict[key]["frame"]

            track_length = len(frames)
            n_axes = 2
            axes_data = np.zeros(shape=(n_axes, track_length))
            axes_data[0] = x
            axes_data[1] = y
            min_t = min(frames) * (1 / self.file_fps)
            max_t = max(frames) * (1 / self.file_fps)
            track_time = max_t - min_t
            time_axis = np.arange(min_t, max_t, (max_t - min_t) / track_length)[:track_length]

            new_track = experimental_tracks.ExperimentalTracks(track_length=track_length,
                                                               track_time=track_time,
                                                               n_axes=n_axes,
                                                               labeling_method=self.file_labeling_method,
                                                               origin_file=self.file_id_db)
            new_track.set_axes_data(axes_data)
            new_track.set_time_axis(time_axis)
            tracks_list.append(new_track)
        return tracks_list
