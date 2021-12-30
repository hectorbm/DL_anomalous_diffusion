from Networks.hurst_exponent_network import HurstExponentNetworkModel
from Tracks.experimental_tracks import ExperimentalTracks
from Tools.db_connection import connect_to_db, disconnect_to_db
from keras import backend as K
from pymongo.errors import CursorNotFound
import numpy as np
# For workers
from worker_config import *
import argparse
worker_mode = False


def train_net(track):
    K.clear_session()
    model_hurst_net = HurstExponentNetworkModel(track_length=track.track_length,
                                                track_time=track.track_time,
                                                fbm_type=track.l2_classified_as,
                                                hiperparams_opt=False)
    model_hurst_net.train_network()
    model_hurst_net.load_model_from_file()
    model_hurst_net.save_model_file_to_db()
    model_hurst_net.save()


def train(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length,
                                        l1_classified_as='fBm', immobile=False)
    count = 1
    for track in tracks:
        networks = HurstExponentNetworkModel.objects(track_length=track.track_length,
                                                     fbm_type=track.l2_classified_as,
                                                     hiperparams_opt=False)
        net_available = False
        for net in networks:
            if net.is_valid_network_track_time(track.track_time):
                net_available = True

        if not net_available:
            if worker_id == (count % num_workers):
                print("Training network for track length:{}, fbm type:{}, and track time:{:.3f}".format(track.track_length,
                                                                                                        track.l2_classified_as,
                                                                                                        track.track_time))
                train_net(track)
        count += 1


def classify(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='fBm', immobile=False)
    if len(tracks):
        networks = HurstExponentNetworkModel.objects(track_length__in=range_track_length, hiperparams_opt=False)
        classified_tracks = {}
        count_classified_tracks = 0
        for track in tracks:
            classified_tracks[str(track.id)] = False

        for net in networks:
            error = np.mean(net.history['val_mae'][-2:])
            if count_classified_tracks < len(tracks):
                if net.load_model_from_file(only_local_files=worker_mode):
                    remaining_tracks = [track for track in tracks if not classified_tracks[str(track.id)] and track.l2_classified_as == net.fbm_type]
                    for track in remaining_tracks:
                        if net.is_valid_network_track_time(track.track_time) and track.track_length == net.track_length:
                            output = net.evaluate_track_input(track)
                            track.set_hurst_exponent(output)
                            track.hurst_mae = error
                            track.save()
                            classified_tracks[str(track.id)] = True
                            count_classified_tracks += 1


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
