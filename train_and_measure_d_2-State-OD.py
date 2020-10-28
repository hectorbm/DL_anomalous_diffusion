from networks.diffusion_coeff_network_model import DiffusionCoefficientNetworkModel
from tracks.experimental_tracks import ExperimentalTracks
from tools.db_connection import connect_to_db, disconnect_to_db
from keras import backend as K
import numpy as np

# For workers
from worker_config import *
import argparse
worker_mode = False


lowerLimitTrackLength = 15


def train_net(track_length, track_time):
    K.clear_session()
    model_d_net = DiffusionCoefficientNetworkModel(track_length=track_length,
                                                   track_time=track_time,
                                                   diffusion_model_range="2-State-OD")
    model_d_net.train_network(batch_size=8)
    model_d_net.load_model_from_file()
    model_d_net.save_model_file_to_db()
    model_d_net.save()


def train(range_track_length):
    for steps in range_track_length:
        tracks = ExperimentalTracks.objects(track_length=steps,
                                            l1_classified_as='2-State-OD')
        count = 1
        for track in tracks:
            for i in range(len(track.seq_initial_frame)):
                if track.track_states[track.seq_initial_frame[i]] == 0 and (
                        track.seq_final_frame[i] - track.seq_initial_frame[i] + 1) > lowerLimitTrackLength:

                    net_available = False
                    networks = DiffusionCoefficientNetworkModel.objects(
                        track_length=(track.seq_final_frame[i] - track.seq_initial_frame[i] + 1),
                        diffusion_model_range=track.l1_classified_as)
                    for net in networks:
                        if net.is_valid_network_track_time(track.seq_res_time[i]):
                            net_available = True

                    if not net_available:
                        if worker_id == (count % num_workers):
                            print('Training for original track length{}, sequence length:{}, sequence time{}'.format(
                                track.track_length, (track.seq_final_frame[i] - track.seq_initial_frame[i] + 1),
                                track.seq_res_time[i]))
                            train_net(track_length=(track.seq_final_frame[i] - track.seq_initial_frame[i] + 1),
                                      track_time=track.seq_res_time[i])
                count += 1


def classify(range_track_length):
    upper_limit = max(range_track_length)
    networks = DiffusionCoefficientNetworkModel.objects(track_length__in=range(lowerLimitTrackLength, upper_limit))
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length,
                                        l1_classified_as='2-State-OD')

    for net in networks:
        K.clear_session()
        if net.load_model_from_file(only_local_files=worker_mode):
            for track in tracks:
                for i in range(len(track.seq_initial_frame)):
                    if track.track_states[track.seq_initial_frame[i]] == 0 and (
                            track.seq_final_frame[i] - track.seq_initial_frame[i] + 1) > lowerLimitTrackLength:
                        if net.is_valid_network_track_time(track.seq_res_time[i]) and (
                                track.seq_final_frame[i] - track.seq_initial_frame[i] + 1) == net.track_length:
                            # Create aux track for analysis
                            seq_track = ExperimentalTracks(
                                track_length=(track.seq_final_frame[i] - track.seq_initial_frame[i] + 1),
                                track_time=track.seq_res_time[i],
                                n_axes=track.n_axes,
                                labeling_method=track.labeling_method,
                                experimental_condition=track.experimental_condition,
                                origin_file=track.origin_file)
                            axes_data = np.zeros(shape=(seq_track.n_axes, seq_track.track_length))
                            axes_data[0] = track.axes_data[str(0)][track.seq_initial_frame[i]:track.seq_final_frame[i] + 1]
                            axes_data[1] = track.axes_data[str(1)][track.seq_initial_frame[i]:track.seq_final_frame[i] + 1]
                            seq_track.set_axes_data(axes_data)
                            seq_track.set_time_axis(
                                np.array(track.time_axis[track.seq_initial_frame[i]:track.seq_final_frame[i] + 1]))
                            seq_track.set_frames(track.frames[track.seq_initial_frame[i]:track.seq_final_frame[i] + 1])

                            output = net.evaluate_track_input(seq_track)
                            track.set_seq_diffusion_coefficient(i, output)
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
    for i in track_length_range:
        print("Training for length:{}".format(i))
        train(range_track_length=[i])
    K.clear_session()
    for i in track_length_range:
        K.clear_session()
        print("Classifying length:{}".format(i))
        classify(range_track_length=[i])
    disconnect_to_db()
