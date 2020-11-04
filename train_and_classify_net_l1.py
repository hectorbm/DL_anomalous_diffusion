from tracks.experimental_tracks import ExperimentalTracks
from networks.l1_network_model import L1NetworkModel
from tools.db_connection import connect_to_db, disconnect_to_db
from keras import backend as K

# For workers
from worker_config import *
import argparse
worker_mode = False


def train_net(track):
    K.clear_session()
    model_l1 = L1NetworkModel(track_length=track.track_length, track_time=track.track_time)
    model_l1.train_network(batch_size=8)
    model_l1.load_model_from_file()
    model_l1.save_model_file_to_db()
    model_l1.save()


def train(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length)
    count = 1
    for track in tracks:
        networks = L1NetworkModel.objects(track_length=track.track_length)
        net_available = False
        for net in networks:
            if net.is_valid_network_track_time(track.track_time):
                net_available = True

        if True:# not net_available:
            if worker_id == (count % num_workers):
                print("Training network for track_length:{} and track_time:{}".format(track.track_length,
                                                                                      track.track_time))
                train_net(track)
        count += 1


def classify(range_track_length):
    print('Classifying tracks')
    networks = L1NetworkModel.objects(track_length__in=range_track_length)
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length)

    for net in networks:
        # In worker mode only check for local files
        if net.load_model_from_file(only_local_files=worker_mode):
            for track in tracks:
                if net.is_valid_network_track_time(track.track_time) and track.track_length == net.track_length:
                    output = net.output_net_to_labels(net.evaluate_track_input(track))
                    track.set_l1_classified(output)

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
