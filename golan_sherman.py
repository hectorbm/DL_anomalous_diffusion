from Tracks.experimental_tracks import ExperimentalTracks
import numpy as np
from Tools.db_connection import connect_to_db, disconnect_to_db


def immobility_filter(threshold, steps_range, label):
    tracks = ExperimentalTracks.objects(track_length__in=steps_range, labeling_method=label)
    immobile_count = 0
    for track in tracks:
        r = 0
        delta_r = []

        # Extract coordinate values from track
        data = np.zeros(shape=(2, track.track_length))
        data[0, :] = track.axes_data["0"]
        data[1, :] = track.axes_data["1"]

        for j in range(track.track_length):
            r = r + np.linalg.norm([data[0, j] - np.mean(data[0, :]), data[1, j] - np.mean(data[1, :])]) ** 2

        for j in range(track.track_length - 1):
            delta_r.append(np.linalg.norm([data[0, j + 1] - data[0, j], data[1, j + 1] - data[1, j]]))

        rad_gir = np.sqrt((1 / track.track_length) * r)
        mean_delta_r = np.mean(delta_r)
        criteria = (rad_gir / mean_delta_r) * np.sqrt(np.pi/2)
        track.immobile = bool(criteria <= threshold)
        if track.immobile:
            immobile_count += 1
        track.save()
    print(label)
    print('Immobile:{:.1f}%\t'.format(100 * immobile_count/len(tracks)), end="")
    print('Mobile:{:.1f}%'.format(100 * (len(tracks)-immobile_count)/len(tracks)))


if __name__ == '__main__':
    connect_to_db()
    # Range
    min_step = 25
    max_step = 900
    steps = list(range(min_step, max_step))
    # Define specific threshold
    defined_threshold_BTX = 1.8
    immobility_filter(threshold=defined_threshold_BTX, steps_range=steps, label='BTX')
    defined_threshold_mAb = 1.8
    immobility_filter(threshold=defined_threshold_mAb, steps_range=steps, label='mAb')

    disconnect_to_db()
