from tools.db_connection import connect_to_db, disconnect_to_db
from tracks.experimental_tracks import ExperimentalTracks, L1_output_categories_labels, EXPERIMENTAL_CONDITIONS, \
    L2_output_categories_labels
import matplotlib.pyplot as plt


def show_tracks_hg():
    connect_to_db()
    tracks = ExperimentalTracks.objects(track_length__in=list(range(20, 850)))
    tracks_len = [track.track_length for track in tracks]
    tracks_len.sort()

    print(len(tracks))
    print(tracks_len[9 * int(len(tracks_len) / 10)])
    plt.hist(x=tracks_len, density=False)
    plt.show()
    disconnect_to_db()


def show_classification_results(tl_range, exp_label, net_name):
    aux = 0
    for exp_cond in EXPERIMENTAL_CONDITIONS:
        tracks = ExperimentalTracks.objects(track_length__in=tl_range, labeling_method=exp_label,
                                            experimental_condition=exp_cond)

        # Count each category
        count = [0 for i in L1_output_categories_labels]
        for track in tracks:
            if net_name == 'L1 Network':
                for i in range(len(L1_output_categories_labels)):
                    if track.l1_classified_as == L1_output_categories_labels[i]:
                        count[i] += 1
            else:
                for i in range(len(L2_output_categories_labels)):
                    if track.l2_classified_as == L2_output_categories_labels[i]:
                        count[i] += 1

        count = [(100 * x) / len(tracks) for x in count]

        plt.bar(x=[(aux + i) for i in range(3)], height=count, width=0.4, align='center',
                color=['firebrick', 'red', 'seagreen'])
        aux += 5

    plt.gca().axes.set_xticks([i for i in range(13)])
    plt.gca().axes.set_xticklabels(['', 'Control', '', '', '', '', 'CDx-Chol', '', '', '', '', 'CDx', ''])
    plt.ylabel('%')
    if exp_label == 'btx':
        exp_label = 'BTX'
    plt.title(exp_label)

    if net_name == 'L1 Network':
        colors = {L1_output_categories_labels[0]: 'firebrick', L1_output_categories_labels[1]: 'red',
                  L1_output_categories_labels[2]: 'seagreen'}
    else:
        colors = {L2_output_categories_labels[0]: 'firebrick', L2_output_categories_labels[1]: 'red',
                  L2_output_categories_labels[2]: 'seagreen'}

    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.show()


if __name__ == '__main__':
    # show_tracks_hg()
    connect_to_db()
    show_classification_results(list(range(25, 100)), 'btx')
    disconnect_to_db()
