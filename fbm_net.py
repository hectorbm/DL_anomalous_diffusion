from Networks.fbm_network import L2NetworkModel
from Tracks.experimental_tracks import ExperimentalTracks
from Tools.db_connection import connect_to_db, disconnect_to_db
from keras import backend as K

# For workers
from worker_config import *
import argparse

worker_mode = False


def train_net(track):
    K.clear_session()
    model_l2 = L2NetworkModel(track_length=track.track_length, track_time=track.track_time)
    model_l2.train_network()
    model_l2.load_model_from_file()
    model_l2.save_model_file_to_db()
    model_l2.save()


def train(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='fBm')
    count = 1
    for track in tracks:
        networks = L2NetworkModel.objects(track_length=track.track_length)
        net_available = False
        for net in networks:
            if net.is_valid_network_track_time(track.track_time):
                net_available = True

        if not net_available:
            if worker_id == (count % num_workers):
                print("Training network for track_length:{} and track_time:{}".format(track.track_length,
                                                                                      track.track_time))
                train_net(track)
        count += 1


def classify(range_track_length):
    print('Classifying tracks')
    networks = L2NetworkModel.objects(track_length__in=range_track_length)
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='fBm')

    for net in networks:
        # In worker mode only check for local files
        if net.load_model_from_file(only_local_files=worker_mode):
            for track in tracks:
                if net.is_valid_network_track_time(track.track_time) and track.track_length == net.track_length:
                    output = net.output_net_to_labels(net.evaluate_track_input(track))
                    track.set_l2_classified(output)

    for track in tracks:
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
    train(range_track_length=track_length_range)
    K.clear_session()
    for i in track_length_range:
        K.clear_session()
        classify(range_track_length=[i])
    disconnect_to_db()
