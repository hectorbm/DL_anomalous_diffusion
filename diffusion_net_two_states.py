from Networks.diffusion_coefficient_network import DiffusionCoefficientNetworkModel
from Tracks.experimental_tracks import ExperimentalTracks
from Tools.db_connection import connect_to_db, disconnect_to_db
from keras import backend as K
import numpy as np
from pymongo.errors import CursorNotFound
# For workers
from worker_config import *
import argparse
worker_mode = False


lowerLimitTrackLength = 15


def train_net(track_length, track_time):
    K.clear_session()
    model_d_net = DiffusionCoefficientNetworkModel(track_length=track_length,
                                                   track_time=track_time,
                                                   diffusion_model_range="2-State-OD",
                                                   hiperparams_opt=False)
    model_d_net.train_network()
    model_d_net.load_model_from_file()
    model_d_net.save_model_file_to_db()
    model_d_net.save()


def train(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='2-State-OD', immobile=False)
    count = 1
    for track in tracks:
        segments = [segment for segment in track.get_brownian_state_segments() if segment['length'] >= lowerLimitTrackLength]

        for segment in segments:
            net_available = False
            networks = DiffusionCoefficientNetworkModel.objects(track_length=segment['length'], hiperparams_opt=False)

            for net in networks:
                if net.is_valid_network_track_time(segment['residence_time']):
                    net_available = True

            if not net_available:
                if worker_id == (count % num_workers):
                    print('Training for original track length:{}, segment length:{}, and segment time:{:.3f}'.format(
                            track.track_length, segment['length'], segment['residence_time']))

                    train_net(track_length=segment['length'], track_time=segment['residence_time'])

            count += 1


def classify(range_track_length):
    upper_limit = max(range_track_length)
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='2-State-OD', immobile=False)
    if len(tracks) > 0:
        networks = DiffusionCoefficientNetworkModel.objects(track_length__in=range(lowerLimitTrackLength, upper_limit),
                                                            hiperparams_opt=False)
        for net in networks:
            try:
                error = np.mean(net.history['val_mae'][-2:])
            except KeyError:
                error = -1
            K.clear_session()
            if net.load_model_from_file(only_local_files=worker_mode):
                for track in tracks:
                    segments = [segment for segment in track.get_brownian_state_segments()
                                if segment['length'] == net.track_length and net.is_valid_network_track_time(segment['residence_time'])]
                    # evaluate segments
                    for segment in segments:
                        sub_track = track.create_track_from_segment(segment)
                        output = net.evaluate_track_input(sub_track)
                        segment['diffusion_coefficient'] = output
                        segment['diffusion_net_error'] = error
                    track.save()


if __name__ == '__main__':
    # Parse params
    parser = argparse.ArgumentParser()
    parser.add_argument("-wm",
                        "--wmode",
                        default=False)
    parser.add_argument("-low",
                        "--rangeLow",
                        type=int,
                        default=25)
    parser.add_argument("-high",
                        "--rangeHigh",
                        type=int,
                        default=100)
    args = parser.parse_args()
    if args.wmode == 'True' and not env_vars_error:
        worker_mode = True
        print('Running in worker mode with worker id:{} and total workers:{}'.format(worker_id,
                                                                                     num_workers))
    else:
        print('Running in standalone mode')
    # Set range to analyze
    track_length_range = list(range(args.rangeLow, args.rangeHigh))
    print('Using range:{} to {}'.format(args.rangeLow, args.rangeHigh))

    connect_to_db()
    # Train, classify and show results
    flag_ex = True
    while flag_ex:
        for i in track_length_range:
            print("Training for length:{}".format(i))
            try:
                train(range_track_length=[i])
                if i == track_length_range[-1]:
                    flag_ex = False
            except CursorNotFound:
                flag_ex = True
        K.clear_session()
    for i in track_length_range:
        K.clear_session()
        print("Classifying length:{}".format(i))
        classify(range_track_length=[i])
    disconnect_to_db()
