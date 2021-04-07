from scipy.io import loadmat
import sys
from Tracks.experimental_tracks import ExperimentalTracks
import numpy as np
from Tools.db_connection import connect_to_db, disconnect_to_db
# Camera parameters
camera_fps = 100


def read_and_load_mat_file():
    mat_data = loadmat('all_tracks_thunder_localizer.mat')
    # Orden en la struct [BTX|mAb] [CDx|Control|CDx-Chol]
    dataset = []
    # Add each label and condition to the dataset
    dataset.append({'label': 'BTX',
                    'exp_cond': 'CDx',
                    'tracks': mat_data['tracks'][0][0]})

    dataset.append({'label': 'BTX',
                    'exp_cond': 'Control',
                    'tracks': mat_data['tracks'][0][1]})

    dataset.append({'label': 'BTX',
                    'exp_cond': 'CDx-Chol',
                    'tracks': mat_data['tracks'][0][2]})

    dataset.append({'label': 'mAb',
                    'exp_cond': 'CDx',
                    'tracks': mat_data['tracks'][1][0]})

    dataset.append({'label': 'mAb',
                    'exp_cond': 'Control',
                    'tracks': mat_data['tracks'][1][1]})

    dataset.append({'label': 'mAb',
                    'exp_cond': 'CDx-Chol',
                    'tracks': mat_data['tracks'][1][2]})

    return dataset


def load_tracks(dataset):
    tracks_to_load = []
    aux = 0
    for elem in dataset:
        print('Loading {} - {}'.format(elem['label'], elem['exp_cond']))
        n_tracks = len(elem['tracks'])
        for i in range(n_tracks):
            track = elem['tracks'][i][0]
            track_instance = create_track_instance(track, elem['label'], elem['exp_cond'])
            tracks_to_load.append(track_instance)

    progress = 0
    for track in tracks_to_load:
        progress += 1
        sys.stdout.write('\r')
        sys.stdout.write('Loading tracks: {:.0f}%'.format(100 * progress / len(tracks_to_load)))
        sys.stdout.flush()
        track.save()

    print('\n{} loaded'.format(len(tracks_to_load)))


def create_track_instance(track, label, exp_cond):
    # Load time axis
    track_time = track[:, 0]
    # Load x, y and convert to nanometers
    track_x = track[:, 1] * 1000
    track_y = track[:, 2] * 1000
    # Trajectory data
    track_length = len(track_time)
    track_duration = track_time[-1] - track_time[0]
    axes_data = np.zeros(shape=(2, track_length))
    axes_data[0] = track_x
    axes_data[1] = track_y
    # Create track instance
    track = ExperimentalTracks(track_length=track_length,
                               track_time=track_duration,
                               n_axes=2,
                               labeling_method=label,
                               experimental_condition=exp_cond)
    track.set_time_axis(track_time)
    track.set_axes_data(axes_data)
    return track


if __name__ == '__main__':
    connect_to_db()
    # my_data = read_and_load_mat_file()
    # load_tracks(my_data)
    range_track = list(range(25, 900))
    
    for label in ['mAb', 'BTX']:
        for exp_cond in ['Control', 'CDx-Chol','CDx']:
            tracks = ExperimentalTracks.objects(track_length__in=range_track, 
                                                labeling_method=label,
                                                experimental_condition=exp_cond,
                                                l1_classified_as='2-State-OD',
                                                immobile=False)        
            count = 0
            od = 0
            brownian = 0
            for track in tracks:
                if len(track.segments) == 1 and track.segments[0]['length'] == track.track_length:
                    count += 1
                    if track.segments[0]['state'] == 0:
                        brownian += 1
                    else:
                        od += 1
            print('Total:{}'.format(len(tracks)))
            print('{} - {}: Count:{}'.format(label, exp_cond, count))
            print('{} - {}: Percentage:{:.3f}%'.format(label, exp_cond, 100 * count/len(tracks)))
            print('{} - {}: Brownian:{}'.format(label, exp_cond, brownian))
            print('{} - {}: OD:{}'.format(label, exp_cond, od))

    disconnect_to_db()
