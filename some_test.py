from tools.db_connection import connect_to_db, disconnect_to_db
from tracks.experimental_tracks import ExperimentalTracks
import matplotlib.pyplot as plt


def show_tracks_hg():
    connect_to_db()
    tracks = ExperimentalTracks.objects(track_length__in=list(range(20, 850)))
    tracks_len = [track.track_length for track in tracks]
    tracks_len.sort()

    print(len(tracks))
    print(tracks_len[9*int(len(tracks_len)/10)])
    plt.hist(x=tracks_len, density=False)
    plt.show()
    disconnect_to_db()


if __name__ == '__main__':
    show_tracks_hg()