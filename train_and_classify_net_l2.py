from networks.l2_network_model import L2NetworkModel
from tracks.experimental_tracks import ExperimentalTracks
from tools.db_connection import connect_to_db, disconnect_to_db
import matplotlib.pyplot as plt
from keras import backend as K


def train_net(track):
    K.clear_session()
    model_l2 = L2NetworkModel(track_length=track.track_length, track_time=track.track_time)
    model_l2.train_network(batch_size=8)
    model_l2.save()


def train(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='fBm')
    for track in tracks:
        networks = L2NetworkModel.objects(track_length=track.track_length)
        net_available = False
        for net in networks:
            if net.is_valid_network_track_time(track.track_time):
                net_available = True

        if not net_available:
            print("Training network for track_length:{} and track_time:{}".format(track.track_length, track.track_time))
            train_net(track)


def classify(range_track_length):
    print('Classifying tracks')
    networks = L2NetworkModel.objects(track_length__in=range_track_length)
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='fBm')
    for net in networks:
        net.load_model_from_file()
        for track in tracks:
            if net.is_valid_network_track_time(track.track_time) and track.track_length == net.track_length:
                output = net.output_net_to_labels(net.evaluate_track_input(track))
                track.set_l2_classified(output)
    for track in tracks:
        track.save()


if __name__ == '__main__':
    track_length_range = list(range(20, 21))
    label = 'mAb'
    exp_cond = 'CDx'

    connect_to_db()
    # Train, classify and show results
    train(range_track_length=track_length_range)
    for i in track_length_range:
        K.clear_session()
        classify(range_track_length=[i])
    disconnect_to_db()
