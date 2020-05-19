from networks.state_detection_network_model import StateDetectionNetworkModel
from tracks.experimental_tracks import ExperimentalTracks

from tools.db_connection import connect_to_db, disconnect_to_db


def train_net(track):
    model_states_net = StateDetectionNetworkModel(track_length=track.track_length, track_time=track.track_time)
    model_states_net.train_network(batch_size=64)
    model_states_net.save()


def train(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='fBm')
    for track in tracks:
        networks = StateDetectionNetworkModel.objects(track_length=track.track_length)
        net_available = False
        for net in networks:
            if net.is_valid_network_track_time(track.track_time):
                net_available = True

        if not net_available:
            print("Training network for track_length:{} and track_time:{}".format(track.track_length, track.track_time))
            train_net(track)


def classify(range_track_length):
    print('Classifying tracks')
    networks = StateDetectionNetworkModel.objects(track_length__in=range_track_length)
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='2-State')
    for net in networks:
        net.load_model_from_file()
        for track in tracks:
            if net.is_valid_network_track_time(track.track_time) and track.track_length == net.track_length:
                output = net.evaluate_track_input(track)
                output = net.convert_output_to_db(output)
                track.set_track_states(output)
    for track in tracks:
        track.save()


def show_results(range_track_length, labeling_method, experimental_condition):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length,
                                        labeling_method=labeling_method,
                                        experimental_condition=experimental_condition,
                                        l1_classified_as='2-State')


if __name__ == '__main__':
    track_length_range = list(range(20, 21))
    label = 'mAb'
    exp_cond = 'CDx'

    connect_to_db()
    # Train, classify and show results
    train(range_track_length=track_length_range)
    classify(range_track_length=track_length_range)
    show_results(range_track_length=track_length_range,
                 labeling_method=label,
                 experimental_condition=exp_cond)

    disconnect_to_db()
