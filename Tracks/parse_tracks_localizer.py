import csv
import os
from experimental_tracks import ExperimentalTracks
import numpy as np
# Camera parameters
px_size = 106
camera_fps = 100

LABELING_METHODS = ['BTX', 'mAb']
EXPERIMENTAL_CONDITIONS = ['Control', 'CDx-Chol', 'CDx']


def get_files_in_path(path_name):
    files = []
    with os.scandir(path_name) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                file_extension = entry.name.split('.')
                if file_extension[len(file_extension) - 1] == "txt":
                    filename = ''.join([path_name, entry.name])
                    files.append(filename)
    return files


def parse(filename, label, exp_cond):
    with open(filename) as fd:
        file = csv.reader(fd, delimiter="\t")
        localizer_headers = ['Particle tracks assembled using Localizer',
                             'IGOR WAVENOTE FOLLOWS',
                             'Localized positions using symmetric 2D Gauss fitting',
                             'IGOR WAVENOTE FOLLOWS',
                             'LOCALIZATION METHOD:0;',
                             'DATA FOLLOWS'
                             ]
        track_header = ['First frame', 'Integrated intensity',
                        'Fitted PSF standard deviation',
                        'X position (pixel)', 'Y position (pixel)',
                        'Background', 'Intensity deviation',
                        'PSF width deviation',
                        'X position deviation',
                        'Y position deviation',
                        'Background deviation',
                        'Number of frames where this emitter is present']
        tracks = []
        n_tracks = 0
        for row in file:
            # Check for single element header
            if len(row) == 1:
                if row[0] not in localizer_headers:
                    single_row = row[0].split(sep=" ")
                    if single_row[0] == 'Contains':
                        n_tracks = int(single_row[1])
                    elif single_row[0] == 'TRACK':
                        x = list()
                        y = list()
                        frames = list()
                    elif single_row[0] == 'END':
                        add_to_file_tracks(frames, tracks, x, y, label, exp_cond)
                    else:
                        raise AssertionError('Error in single element header{}'.format(single_row))
            # Check for track header and localizations
            if len(row) == 12:
                if row != track_header:
                    x.append(float(row[3]) * px_size)
                    y.append(float(row[4]) * px_size)
                    frames.append(int(row[0]))

    if len(tracks) == n_tracks:
        print('Added:{} tracks'.format(n_tracks))
    else:
        raise AssertionError('Error in number of tracks:{}, array contains:{}'.format(n_tracks, len(tracks)))
    return tracks


def add_to_file_tracks(frames, tracks, x, y, label, exp_cond):
    track_length = len(frames)
    axes_data = np.zeros(shape=(2, track_length))
    axes_data[0] = x
    axes_data[1] = y
    min_t = frames[0] * (1 / camera_fps)
    max_t = frames[-1] * (1 / camera_fps)
    track_time = max_t - min_t
    time_axis = np.arange(min_t, max_t, (max_t - min_t) / track_length)[:track_length]
    track = ExperimentalTracks(track_length=track_length,
                               track_time=track_time,
                               n_axes=2,
                               labeling_method=label,
                               experimental_condition=exp_cond)
    track.set_frames(frames)
    track.set_axes_data(axes_data)
    track.set_time_axis(time_axis)
    tracks.append(track)
