from networks.state_detection_network_model import StateDetectionNetworkModel
from tracks.experimental_tracks import ExperimentalTracks
from tools.db_connection import connect_to_db, disconnect_to_db
from keras import backend as K


def train_net(track):
    K.clear_session()
    model_states_net = StateDetectionNetworkModel(track_length=track.track_length, track_time=track.track_time)
    model_states_net.train_network(batch_size=8)
    model_states_net.save()


def train(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length, l1_classified_as='2-State-OD')
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
                track.save()
    for track in tracks:
        track.compute_sequences_length()
        track.compute_sequences_res_time()
        track.compute_confinement_regions()
        track.save()


if __name__ == '__main__':
    track_length_range = list(range(20, 21))
    label = 'mAb'
    exp_cond = 'CDx'

    connect_to_db()
    # Train, classify and show results
    train(range_track_length=track_length_range)
    K.clear_session()
    for i in track_length_range:
        K.clear_session()
        classify(range_track_length=[i])

    disconnect_to_db()
