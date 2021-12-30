from Tools.db_connection import connect_to_db, disconnect_to_db
from Tracks.experimental_tracks import ExperimentalTracks, L1_output_categories_labels, EXPERIMENTAL_CONDITIONS, \
    L2_output_categories_labels
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from Networks.classification_network import L1NetworkModel
from Networks.fbm_network import L2NetworkModel
from Networks.states_detection_network import StateDetectionNetworkModel
from Networks.hurst_exponent_network import HurstExponentNetworkModel
from Networks.diffusion_coefficient_network import DiffusionCoefficientNetworkModel
import keras.backend as K
from scipy.optimize import curve_fit
from scipy import stats
# For PDF
from reliability.Fitters import Fit_Exponential_1P, Fit_Normal_2P
from reliability.Other_functions import histogram


def get_classification_error(steps_range, exp_label, exp_cond, net_name):
    if net_name == 'L1 Network':
            tracks = ExperimentalTracks.objects(track_length__in=steps_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, immobile=False)
    else:
        tracks = ExperimentalTracks.objects(track_length__in=steps_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, l1_classified_as='fBm', immobile=False)

    classification_accuracy = []
    for track in tracks:
        if net_name == 'L1 Network':
            classification_accuracy.append(track.l1_error)
        elif net_name == 'L2 Network':
            classification_accuracy.append(track.l2_error)
        else: 
            raise ValueError

    lower_x = np.percentile(classification_accuracy, 5)
    # histogram(classification_accuracy, c='orange', white_above=lower_x)
    # plt.axvline(x=lower_x,c='black',alpha=0.7,linestyle='dotted')
    # plt.ylabel('Frequency', fontsize=16)
    # plt.xlabel('Classification accuracy', fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.show()
    return lower_x


# Performance plots
def validation_set_confusion_matrix(net_name):
    for i in [25, 50, 100]:
        if net_name == 'L1 Network':
            net = L1NetworkModel.objects(track_length=i, hiperparams_opt=False).order_by('track_time')[0]
        elif net_name == 'L2 Network':
            net = L2NetworkModel.objects(track_length=i, hiperparams_opt=False).order_by('track_time')[0]
        elif net_name == 'Segmentation Network':
            net = StateDetectionNetworkModel.objects(track_length=i, hiperparams_opt=False).order_by('track_time')[0]
        net.load_model_from_file()
        net.validate_test_data_accuracy(n_axes=2)


def net_mae(min_steps, max_steps, net_name):
    if net_name == 'Hurst Exponent Network':
        diffusion_range = ["Subdiffusive", "Brownian", "Superdiffusive"]
    elif net_name == 'Diffusion Coefficient Network':
        diffusion_range = ['Brownian']

    for fbm_type in diffusion_range:
        error_arr = []
        x = []
        for i in range(min_steps, max_steps):
            try:
                if net_name == 'Hurst Exponent Network':
                    net = HurstExponentNetworkModel.objects(track_length=i, fbm_type=fbm_type, hiperparams_opt=False).order_by('track_time')[0]
                elif net_name == 'Diffusion Coefficient Network':
                    net = DiffusionCoefficientNetworkModel.objects(track_length=i, hiperparams_opt=False).order_by('track_time')[0]

                error = np.mean(net.history['val_mae'][-2:])
                error_arr.append(error)
                x.append(i)
            except IndexError:
                print('Network not available for #{} steps'.format(i))

        plt.plot(x, error_arr)
        plt.xlabel('Steps', fontsize=16)
        plt.ylabel('MAE', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        if net_name == 'Hurst Exponent Network':
            plt.title(fbm_type, fontsize=16)
        elif net_name == 'Diffusion Coefficient Network':
            plt.title('Diffusion Coefficient Network', fontsize=16)

        plt.show()


def net_mae_histogram(min_steps, max_steps,net_name):
    error_arr = []
    for i in range(min_steps, max_steps):
        try:
            net = DiffusionCoefficientNetworkModel.objects(track_length=i, hiperparams_opt=False).order_by('track_time')[0]
            error = np.mean(net.history['val_mae'][-4:])
            error_arr.append(error)

        except IndexError:
            print('Network not available for #{} steps'.format(i))

    histogram(error_arr, bins=15)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('MAE', fontsize=16)
    plt.xticks([round(i, 2) for i in list(np.linspace(0.01, 0.05, 5))], fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


# Results plots
def show_classification_results(tl_range, exp_label, net_name):
    aux = 0
    for exp_cond in EXPERIMENTAL_CONDITIONS:
        # Get classification error
        pc = 1 - get_classification_error(tl_range, exp_label, exp_cond, net_name)

        if net_name == 'L1 Network':
            tracks = ExperimentalTracks.objects(track_length__in=tl_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, immobile=False)
        else:
            tracks = ExperimentalTracks.objects(track_length__in=tl_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, l1_classified_as='fBm', immobile=False)

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

        # Compute error limits
        count_n = count
        error_y0 = (100 * pc * count[0]/len(tracks), 100 * pc * (count[1]+count[2])/len(tracks))
        error_y1 = (100 * pc * count[1]/len(tracks), 100 * pc * (count[0]+count[2])/len(tracks))
        error_y2 = (100 * pc * count[2]/len(tracks), 100 * pc * (count[0]+count[1])/len(tracks))
        count = [(100 * x) / len(tracks) for x in count]
        error_y = [[error_y0[0], error_y1[0], error_y2[0]], [error_y0[1], error_y1[1], error_y2[1]]]

        # For data tables
        print('Network:{}, label:{}, condition:{}'.format(net_name, exp_label, exp_cond))
        if net_name == 'L1 Network':
            print('{}, {}, {}'.format(L1_output_categories_labels[0], L1_output_categories_labels[1], L1_output_categories_labels[2]))
        else:
            print('{}, {}, {}'.format(L2_output_categories_labels[0], L2_output_categories_labels[1], L2_output_categories_labels[2]))
        print('{}, {}, {}'.format(count[0], count[1], count[2]))
        print(count_n)
        print(error_y)

        # Plot bars
        plt.bar(x=[(aux + i) for i in range(3)], height=count, width=0.6, align='center',
                color=['firebrick', 'orangered', 'dodgerblue'], yerr=error_y)
        aux += 5

    plt.gca().axes.set_xticks([i for i in range(13)])
    plt.gca().axes.set_xticklabels(['', 'Control', '', '', '', '', 'CDx-Chol', '', '', '', '', 'CDx', ''], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('%', fontsize=16)
    if exp_label == 'BTX':
        exp_label = 'BTX'
    plt.title(exp_label, fontsize=16)

    if net_name == 'L1 Network':
        colors = {L1_output_categories_labels[0]: 'firebrick', L1_output_categories_labels[1]: 'orangered',
                  L1_output_categories_labels[2]: 'dodgerblue'}
    else:
        colors = {L2_output_categories_labels[0]: 'firebrick', L2_output_categories_labels[1]: 'orangered',
                  L2_output_categories_labels[2]: 'dodgerblue'}

    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]

    if net_name == 'L1 Network' and label == 'mAb':
        plt.legend(handles, ['fBm', 'CTRW', 'two-state'], bbox_to_anchor=(1.04, 1), borderaxespad=0, fontsize=14)
    elif net_name == 'L2 Network' and label == 'mAb':
        plt.legend(handles, ['fBm subdiffusive', 'fBm Brownian', 'fBm superdiffusive'], bbox_to_anchor=(1.04, 1), borderaxespad=0, fontsize=14)

    plt.rcParams['lines.color'] = 'b'
    plt.rcParams['lines.linewidth'] = 3
    plt.show()


def show_hurst_results(range_steps, label):
    l2_categories = ['Subdiffusive', 'Brownian', 'Superdiffusive']

    for category in l2_categories:
        aux = 1
        for exp_cond in EXPERIMENTAL_CONDITIONS:
            print('Hurst Exponent, Label:{}, condition:{} range:{}'.format(label, exp_cond, category))

            tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                                experimental_condition=exp_cond, l1_classified_as='fBm',
                                                l2_classified_as=category, immobile=False)
            hurst_exp_values = [track.hurst_exponent_fbm for track in tracks]
            
            print(hurst_exp_values)


def show_residence_time(range_steps, label):
    for state in [0, 1]:
        for exp_cond in EXPERIMENTAL_CONDITIONS:
            tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                                experimental_condition=exp_cond, l1_classified_as='2-State-OD',immobile=False)
            # Compute residence time values
            res_time_values = []
            for track in tracks:
                if state == 0:
                    segments = track.get_brownian_state_segments()
                else:
                    segments = track.get_od_state_segments()
                for segment in segments:
                    if track.segments[0]['length'] != track.track_length:
                        res_time_values.append(segment['residence_time'])
            

            print('Residence time, state:{} , Label:{}, condition:{}'.format(state, label, exp_cond))          
            print(res_time_values)


def show_confinement_area(range_steps, label):
    for exp_cond in EXPERIMENTAL_CONDITIONS:
        tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                            experimental_condition=exp_cond, l1_classified_as='2-State-OD', immobile=False)
        conf_area_values = []
        for track in tracks:
            segments = track.get_od_state_segments()
            for segment in segments:
                if track.segments[0]['length'] != track.track_length:
                    conf_area_values.append(segment['confinement_area'] * (0.001 ** 2))


        print('Conf area, Label:{}, condition:{}'.format(label, exp_cond)) 
        print(conf_area_values)
        
def show_transitions(range_steps, label):
    for exp_cond in EXPERIMENTAL_CONDITIONS:
        tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                            experimental_condition=exp_cond, l1_classified_as='2-State-OD', immobile=False)
        OD_to_Brownian = []
        Brownian_to_OD = []

        for track in tracks:
            if track.transitions['OD_to_Brownian'] > 0:
                OD_to_Brownian.append(track.transitions['OD_to_Brownian'])
            if track.transitions['Brownian_to_OD'] > 0:
                Brownian_to_OD.append(track.transitions['Brownian_to_OD'])
        
        print('OD to Brownian, Label:{}, condition:{}'.format(label, exp_cond)) 
        print(OD_to_Brownian)
        print('Brownian to OD, Label:{}, condition:{}'.format(label, exp_cond)) 
        print(Brownian_to_OD)

def show_diffusion_results_brownian(range_steps, label):
    for exp_cond in EXPERIMENTAL_CONDITIONS:
        tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                            experimental_condition=exp_cond, l1_classified_as='fBm',
                                            l2_classified_as='Brownian', immobile=False)
        diffusion_coefficient_values = []
        aux = 0
        for track in tracks:
            if track.diffusion_coefficient_brownian > 0:
                diffusion_coefficient_values.append(track.diffusion_coefficient_brownian)
            else:
                aux += 1 
        
        print('Diffusion coefficient, Label:{}, condition:{}'.format(label, exp_cond)) 
        print('Bad training:{}/{}'.format(aux,len(tracks)))

        if len(diffusion_coefficient_values) > 2:
            normFit = Fit_Normal_2P(failures=diffusion_coefficient_values)
            plt.show()

            histogram(diffusion_coefficient_values)
            normFit.distribution.PDF()
            mean = normFit.mu
            stdev = normFit.sigma
            plt.title('{} - {}\n{}:{:.2f}, {}:{:.2f}'.format(label, exp_cond, r'$\mu$', mean, r'$\sigma$',stdev), fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.xlabel('Diffusion coefficient {}m²'.format(r'$\mu$'), fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()


def show_diffusion_coefficient_two_state(range_steps, label):
    keyerror = 0
    for exp_cond in EXPERIMENTAL_CONDITIONS:
        tracks = ExperimentalTracks.objects(track_length__in=range_steps, labeling_method=label,
                                                experimental_condition=exp_cond, l1_classified_as='2-State-OD',
                                                immobile=False)
        diffusion_coefficient_values = []
        for track in tracks:
            segments  = track.get_brownian_state_segments()
            for segment in segments:
                try:
                    d = segment['diffusion_coefficient']
                    if 0.2>= d >= 0.05:
                        diffusion_coefficient_values.append(d)
                except KeyError:
                    keyerror += 1 
        print('Diffusion coefficient, Label:{}, condition:{}'.format(label, exp_cond)) 
        print(diffusion_coefficient_values)


if __name__ == '__main__':
    connect_to_db()
    min_steps = 25
    max_steps = 900
    exp_labels = ['mAb', 'BTX']
    net = 'L1 Network'
    range_steps = list(range(min_steps, max_steps))
    
    # Performance plots
    # validation_set_confusion_matrix(net_name='L2 Network')

    # net_mae(min_steps, max_steps, 'Hurst Exponent Network')
    net_mae_histogram(min_steps, max_steps, 'Diffusion Coefficient Network')
    # for label in exp_labels:
    #     show_classification_results(range_steps, label, net)

    # net = 'L2 Network'
    # for label in exp_labels:
    #     show_classification_results(range_steps, label, net)

    # for label in exp_labels:
    #     # show_hurst_results(range_steps, label)
    #     # show_residence_time(range_steps, label)
    #     # show_confinement_area(range_steps,label)
    #     # show_transitions(range_steps,label)
    #     # show_diffusion_results_brownian(range_steps,label)
    #     show_diffusion_coefficient_two_state(range_steps,label)
    disconnect_to_db()
