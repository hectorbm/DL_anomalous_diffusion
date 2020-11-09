from Networks.classification_network import L1NetworkModel
from Networks.fbm_network import L2NetworkModel
from Networks.states_detection_network import StateDetectionNetworkModel
from Networks.hurst_exponent_network import HurstExponentNetworkModel
from Networks.diffusion_coefficient_network import DiffusionCoefficientNetworkModel
from Tools.db_connection import connect_to_db, disconnect_to_db
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


def create_model(net_name):
    if net_name == 'L1Network':
        network = L1NetworkModel(track_length=track_length,
                                 track_time=track_time,
                                 hiperparams_opt=True)
    elif net_name == 'L2Network':
        network = L2NetworkModel(track_length=track_length,
                                 track_time=track_time,
                                 hiperparams_opt=True)
    elif net_name == 'StateDetectionNetwork':
        network = StateDetectionNetworkModel(track_length=track_length,
                                             track_time=track_time,
                                             hiperparams_opt=True)
    elif net_name == 'HurstExponentNetwork':
        network = HurstExponentNetworkModel(track_length=track_length,
                                            track_time=track_time,
                                            fbm_type=fbm_type_net,
                                            hiperparams_opt=True)
    else:
        network = DiffusionCoefficientNetworkModel(track_length=track_length,
                                                   track_time=track_time,
                                                   diffusion_model_range='Brownian',
                                                   hiperparams_opt=True)

    return network


def scan_params(net_name):
    # Stack names and lists position
    if len(analysis_params) > 0:
        stack_names = [k for k, v in analysis_params.items()]
        stack = [0 for i in stack_names]
        tos = len(stack) - 1
        analysis_ended = False
        increasing = True

        # Compute and print number of combinations
        number_of_combinations = len(analysis_params[stack_names[0]])
        for i in range(1, len(stack_names)):
            number_of_combinations *= len(analysis_params[stack_names[i]])
        print("Total of combinations:{}".format(number_of_combinations))

        # Run the analysis
        while not analysis_ended:
            if tos == (len(stack) - 1) and stack[tos] < len(analysis_params[stack_names[tos]]):
                K.clear_session()
                network = create_model(net_name)
                for i in range(len(stack_names)):
                    network.net_params[stack_names[i]] = analysis_params[stack_names[i]][stack[i]]
                print('Evaluating params: {}'.format(network.net_params))

                network.train_network(batch_size=network.net_params['batch_size'])
                network.load_model_from_file()
                network.save_model_file_to_db()
                network.save()
                stack[tos] += 1
            elif tos == (len(stack) - 1) and stack[tos] == len(analysis_params[stack_names[tos]]):
                stack[tos] = 0
                tos -= 1
                increasing = False

            elif 0 < tos < (len(stack) - 1) and increasing:
                tos += 1
                increasing = True
            elif 0 < tos < (len(stack) - 1) and not increasing and stack[tos] + 1 <= len(
                    analysis_params[stack_names[tos]]) - 1:
                stack[tos] += 1
                tos += 1
                increasing = True
            elif 0 < tos < (len(stack) - 1) and not increasing and stack[tos] + 1 > len(
                    analysis_params[stack_names[tos]]) - 1:
                stack[tos] = 0
                tos -= 1
                increasing = False
            elif tos == 0 and not increasing and stack[tos] + 1 < len(analysis_params[stack_names[tos]]):
                stack[tos] += 1
                tos += 1
                increasing = True
            else:
                analysis_ended = True


def get_params(net_name):
    net_params = {
        'L1Network': L1NetworkModel.analysis_params,
        'L2Network': L2NetworkModel.analysis_params,
        'StateDetectionNetwork': StateDetectionNetworkModel.analysis_params,
        'HurstExponentNetwork': HurstExponentNetworkModel.analysis_params,
        'DiffusionCoefficientNetwork': DiffusionCoefficientNetworkModel.analysis_params
    }
    params = net_params[net_name]
    return params


def sort_results(e):
    return e[0]


def my_filter(e):
    if e[1] > 0.2:
        return False
    else:
        return True


def plot_analysis(net_name):
    first_reduction = 35
    second_reduction = 10

    if net_name == 'L1Network':
        networks = L1NetworkModel.objects(hiperparams_opt=True, track_length=track_length, track_time=track_time)
    elif net_name == 'L2Network':
        networks = L2NetworkModel.objects(hiperparams_opt=True, track_length=track_length, track_time=track_time)
    elif net_name == 'StateDetectionNetwork':
        networks = StateDetectionNetworkModel.objects(track_length=track_length, track_time=track_time,
                                                      hiperparams_opt=True)
    elif net_name == 'HurstExponentNetwork':
        networks = HurstExponentNetworkModel.objects(track_length=track_length,
                                                     track_time=track_time,
                                                     fbm_type=fbm_type_net,
                                                     hiperparams_opt=True)
    else:
        networks = DiffusionCoefficientNetworkModel.objects(track_length=track_length,
                                                            track_time=track_time,
                                                            diffusion_model_range='Brownian',
                                                            hiperparams_opt=True)

    print("Total training results:{}".format(len(networks)))
    for network in networks:
        epochs = [(i + 1) for i in range(len(network.history['val_loss']))]
        plt.plot(epochs, network.history['val_loss'])
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    plt.show()

    # Delete outliers if needed, modify stdev limit if required
    results = [[np.mean(network.history['val_loss']),
                np.std(network.history['val_loss']),
                network.id]
               for network in networks]
    results = [res for res in results if my_filter(res)]
    print("Now using:{}".format(len(results)))
    results = [res[2] for res in results]
    networks = [network for network in networks if network.id in results]
    # Show results after filter
    for network in networks:
        epochs = [(i + 1) for i in range(len(network.history['val_loss']))]
        plt.plot(epochs, network.history['val_loss'])
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    plt.show()

    # First reduction
    print("Set min:", end="")
    min_epoch = int(input())
    print("Set max:", end="")
    max_epoch = int(input())

    # Compute mean and stdev in defined range
    results = [[np.mean(network.history['val_loss'][min_epoch:max_epoch + 1]),
                np.std(network.history['val_loss'][min_epoch:max_epoch + 1]),
                network.id]
               for network in networks]
    # Sort all the values
    results.sort(key=sort_results, reverse=False)
    best_results = results[0:first_reduction]
    results = [res[2] for res in results]
    networks = [network for network in networks if network.id in results[0:first_reduction]]

    # Show after reduction
    for network in networks:
        epochs = [(j + 1) for j in range(len(network.history['val_loss']))]
        plt.plot(epochs, network.history['val_loss'])
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    plt.show()
    # Second reduction
    print("Set min:", end="")
    min_epoch = int(input())
    print("Set max:", end="")
    max_epoch = int(input())

    # Compute mean and stdev in defined range
    results = [[np.mean(network.history['val_loss'][min_epoch:max_epoch]),
                np.std(network.history['val_loss'][min_epoch:max_epoch]),
                network.id]
               for network in networks]
    # Sort all the values
    results.sort(key=sort_results, reverse=False)

    best_results = results[0:second_reduction]
    results = [res[2] for res in results]
    networks = [network for network in networks if network.id in results[0:second_reduction]]

    i = 0
    for network in networks:
        epochs = [(j + 1) for j in range(len(network.history['val_loss']))]
        plt.plot(epochs, network.history['val_loss'])
        print('{}, {} ,{}'.format(network.params_training, best_results[i][0], best_results[i][1]))
        i += 1
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')
    plt.show()

    print("Training loss results")
    i = 0
    for network in networks:
        epochs = [(j + 1) for j in range(len(network.history['loss']))]
        plt.plot(epochs, network.history['loss'])
        i += 1
    plt.ylabel('Training loss')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    track_length = 25
    track_time = 0.5
    fbm_type_net = 'Brownian'

    net = 'StateDetectionNetwork'

    connect_to_db()
    analysis_params = get_params(net)
    # scan_params(net)
    plot_analysis(net)
    disconnect_to_db()
