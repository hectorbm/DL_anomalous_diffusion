from tracks.experimental_tracks import ExperimentalTracks
from networks.l1_network_model import L1NetworkModel
from tools.db_connection import connect_to_db, disconnect_to_db


def train_net(track):
    model_l1 = L1NetworkModel(track_length=track.track_length, track_time=track.track_time)
    model_l1.train_network(batch_size=64)
    model_l1.save()


def train(range_track_length):
    tracks = ExperimentalTracks.objects(track_length__in=range_track_length)
    for track in tracks:
        networks = L1NetworkModel.objects(track_length=track.track_length)
        net_available = False
        for net in networks:
            if net.is_valid_network_track_time(track.track_time):
                net_available = True

        if not net_available:
            print("training network for track_length:{} and track_time:{}".format(track.track_length, track.track_time))
            train_net(track)


def classify(range_track_length):
    pass


if __name__ == '__main__':
    track_length_range = list(range(20, 21))
    connect_to_db()
    train(range_track_length=track_length_range)
    #classify(track_length_range=track_length_range)
    disconnect_to_db()
