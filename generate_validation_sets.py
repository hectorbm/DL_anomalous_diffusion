import math
import pickle

from networks.l1_network_model import L1NetworkModel
from networks.l2_network_model import L2NetworkModel
from networks.state_detection_network_model import StateDetectionNetworkModel
from networks.hurst_exp_network_model import HurstExponentNetworkModel

from networks.generators import generate_batch_of_samples_l1, generate_batch_of_samples_l2, \
    generate_batch_of_samples_state_net, generate_batch_diffusion_coefficient_net, generate_batch_hurst_net


# TODO: UPDATE TO NEW DATASET SIZE, CHECK FOR ERRORS!


def classification_net_val_data():
    x_val = []
    y_val = []
    print("Simulating samples Classification Net")
    out, label = generate_batch_of_samples_l1(L1NetworkModel.net_params['validation_set_size'], length, time)
    x_val.append(out)
    y_val.append(label)

    with open('networks/val_data/classification_net/x_val_len_{}_time_{}.pkl'.format(length, time),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/classification_net/y_val_len_{}_time_{}.pkl'.format(length, time),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


def fbm_net_val_data():
    x_val = []
    y_val = []
    print("Simulating samples fBm Net")
    out, label = generate_batch_of_samples_l2(L2NetworkModel.net_params['validation_set_size'], length, time)
    x_val.append(out)
    y_val.append(label)
    with open('networks/val_data/fbm_net/x_val_len_{}_time_{}.pkl'.format(length, time),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/fbm_net/y_val_len_{}_time_{}.pkl'.format(length, time),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


def states_net_val_data():
    x_val = []
    y_val = []
    print("Simulating samples states net")
    out, label = generate_batch_of_samples_state_net(StateDetectionNetworkModel.net_params['validation_set_size'],
                                                     length, time)
    x_val.append(out)
    y_val.append(label)
    with open('networks/val_data/states_net/x_val_len_{}_time_{}.pkl'.format(length, time),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/states_net/y_val_len_{}_time_{}.pkl'.format(length, time),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


def diffusion_coefficient_net_val_data():
    x_val = []
    y_val = []
    # TODO: Change for new Net!!
    # for i in range(math.ceil(4800 / batch)):
    #     out, label = generate_batch_diffusion_coefficient_net(batch, model_range, length, time)
    #     x_val.append(out)
    #     y_val.append(label)
    # with open('networks/val_data/diffusion_net/x_val_len_{}_time_{}_batch_{}_range_{}.pkl'.format(length, time, batch,
    #                                                                                               model_range),
    #           'wb') as x_val_data:
    #     pickle.dump(x_val, x_val_data)
    # with open('networks/val_data/diffusion_net/y_val_len_{}_time_{}_batch_{}_range_{}.pkl'.format(length, time, batch,
    #                                                                                               model_range),
    #           'wb') as y_val_data:
    #     pickle.dump(y_val, y_val_data)


def hurst_net_val_data():
    x_val = []
    y_val = []
    print("Simulating samples Hurst Net")
    out, label = generate_batch_hurst_net(HurstExponentNetworkModel.net_params['validation_set_size'], fbm_type, length,
                                          time)
    x_val.append(out)
    y_val.append(label)
    with open('networks/val_data/hurst_net/x_val_len_{}_time_{}_fbm_type_{}.pkl'.format(length, time, fbm_type),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/hurst_net/y_val_len_{}_time_{}_fbm_type_{}.pkl'.format(length, time, fbm_type),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


if __name__ == '__main__':
    length = 50
    time = 1.0
    batch = 32
    model_range = '2-State-OD'
    fbm_type = 'Subdiffusive'

    # Remove comment to create the data
    # classification_net_val_data()

    fbm_net_val_data()

    states_net_val_data()

    hurst_net_val_data()
