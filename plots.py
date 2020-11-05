from tools.db_connection import connect_to_db, disconnect_to_db
from tracks.experimental_tracks import ExperimentalTracks, L1_output_categories_labels, EXPERIMENTAL_CONDITIONS, \
    L2_output_categories_labels
import matplotlib.pyplot as plt
import numpy as np


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

        if net_name == 'L1 Network':
            tracks = ExperimentalTracks.objects(track_length__in=tl_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond)
        else:
            tracks = ExperimentalTracks.objects(track_length__in=tl_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, l1_classified_as='fBm')

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
    if exp_label == 'BTX':
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


def show_hurst_results(range_steps, label):
    l2_categories = ['Subdiffusive', 'Brownian', 'Superdiffusive']

    for category in l2_categories:
        aux = 1
        for exp_cond in EXPERIMENTAL_CONDITIONS:
            tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                                experimental_condition=exp_cond, l1_classified_as='fBm',
                                                l2_classified_as=category)
            hurst_exp_values = [track.hurst_exponent_fbm for track in tracks]
            plt.violinplot(hurst_exp_values, widths=1.5, positions=[aux], showextrema=False, showmeans=False)
            plt.scatter(x=np.random.uniform(low=aux - 0.35, high=aux + 0.35, size=len(hurst_exp_values)),
                        y=hurst_exp_values, s=20, alpha=0.6)
            stats_values = [np.mean(hurst_exp_values), np.mean(hurst_exp_values) - np.std(hurst_exp_values),
                            np.mean(hurst_exp_values) + np.std(hurst_exp_values)]
            plt.scatter(x=[aux for i in range(3)], y=stats_values, color='black', marker='+', s=100, alpha=1)
            plt.plot([aux for i in range(3)], stats_values, color='black')
            aux = aux + 3

        if label == 'BTX':
            plt.title('BTX - {}'.format(category))
        else:
            plt.title('{} - {}'.format(label, category))
        plt.gca().axes.set_xticks([i for i in range(8)])
        plt.gca().axes.set_xticklabels(['', 'Control', '', '', 'CDx-Chol', '', '', 'CDx'])
        plt.ylabel('Hurst Exponent')
        plt.show()


def show_residence_time(range_steps, label):
    for state in [0, 1]:
        aux = 1
        for exp_cond in EXPERIMENTAL_CONDITIONS:
            tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                                experimental_condition=exp_cond, l1_classified_as='2-State-OD')
            res_time_values = []
            for track in tracks:
                track_res_time = track.get_res_time_state(state)
                for res_time in track_res_time:
                    res_time_values.append(res_time)
            plt.violinplot(res_time_values, widths=1.5, positions=[aux], showextrema=False, showmeans=False)
            plt.scatter(x=np.random.uniform(low=aux - 0.35, high=aux + 0.35, size=len(res_time_values)),
                        y=res_time_values, s=20, alpha=0.6)
            stats_values = [np.mean(res_time_values), np.mean(res_time_values) - np.std(res_time_values),
                            np.mean(res_time_values) + np.std(res_time_values)]
            plt.scatter(x=[aux for i in range(3)], y=stats_values, color='black', marker='+', s=100, alpha=1)
            plt.plot([aux for i in range(3)], stats_values, color='black')
            aux = aux + 3

        if label == 'BTX' and state == 0:
            plt.title('BTX - Free diffusion state')
        elif label == 'BTX' and state == 1:
            plt.title('BTX - Confined state')
        elif label == 'mAb' and state == 0:
            plt.title('mAb - Free diffusion state')
        else:
            plt.title('mAb - Confined state')
        plt.gca().axes.set_xticks([i for i in range(8)])
        plt.gca().axes.set_xticklabels(['', 'Control', '', '', 'CDx-Chol', '', '', 'CDx'])
        plt.ylabel('Residence Time')
        plt.show()


def show_confinement_area(range_steps, label):
    aux = 1
    for exp_cond in EXPERIMENTAL_CONDITIONS:
        tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                            experimental_condition=exp_cond, l1_classified_as='2-State-OD')
        conf_area_values = []
        for track in tracks:
            for area in track.confinement_regions_area:
                if area > 0:
                    conf_area_values.append(area * (0.001 ** 2))

        plt.violinplot(conf_area_values, widths=1.5, positions=[aux], showextrema=False, showmeans=False)
        plt.scatter(x=np.random.uniform(low=aux - 0.35, high=aux + 0.35, size=len(conf_area_values)),
                    y=conf_area_values, s=20, alpha=0.6)
        stats_values = [np.mean(conf_area_values), np.mean(conf_area_values) - np.std(conf_area_values),
                        np.mean(conf_area_values) + np.std(conf_area_values)]
        plt.scatter(x=[aux for i in range(3)], y=stats_values, color='black', marker='+', s=100, alpha=1)
        plt.plot([aux for i in range(3)], stats_values, color='black')
        aux = aux + 3

    if label == 'BTX':
        plt.title('BTX - Confined state')
    else:
        plt.title('mAb - Confined state')
    plt.gca().axes.set_xticks([i for i in range(8)])
    plt.gca().axes.set_xticklabels(['', 'Control', '', '', 'CDx-Chol', '', '', 'CDx'])
    plt.ylabel('Confinement area {}m²'.format(r'$\mu$'))
    plt.show()


if __name__ == '__main__':
    connect_to_db()
    min_steps = 25
    max_steps = 100
    exp_labels = ['mAb', 'BTX']
    net = 'L2 Network'

    range_steps = list(range(min_steps, max_steps))
    # for label in exp_labels:
    #     show_classification_results(range_steps, label, net)

    for label in exp_labels:
        show_hurst_results(range_steps, label)
    #     show_residence_time(range_steps, label)
    #     show_confinement_area(range_steps,label)
    disconnect_to_db()
