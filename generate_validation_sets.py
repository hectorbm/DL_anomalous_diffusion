import math
import pickle

from networks.generators import generate_batch_of_samples_l1, generate_batch_of_samples_l2, \
    generate_batch_of_samples_state_net, generate_batch_diffusion_coefficient_net, generate_batch_hurst_net


def classification_net_val_data():
    x_val = []
    y_val = []
    for i in range(math.ceil(4800 / batch)):
        out, label = generate_batch_of_samples_l1(batch, length, time)
        x_val.append(out)
        y_val.append(label)
    with open('networks/val_data/classification_net/x_val_len_{}_time_{}_batch_{}.pkl'.format(length, time, batch),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/classification_net/y_val_len_{}_time_{}_batch_{}.pkl'.format(length, time, batch),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


def fbm_net_val_data():
    x_val = []
    y_val = []
    for i in range(math.ceil(4800 / batch)):
        out, label = generate_batch_of_samples_l2(batch, length, time)
        x_val.append(out)
        y_val.append(label)
    with open('networks/val_data/fbm_net/x_val_len_{}_time_{}_batch_{}.pkl'.format(length, time, batch),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/fbm_net/y_val_len_{}_time_{}_batch_{}.pkl'.format(length, time, batch),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


def states_net_val_data():
    x_val = []
    y_val = []
    for i in range(math.ceil(4800 / batch)):
        out, label = generate_batch_of_samples_state_net(batch, length, time)
        x_val.append(out)
        y_val.append(label)
    with open('networks/val_data/states_net/x_val_len_{}_time_{}_batch_{}.pkl'.format(length, time, batch),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/states_net/y_val_len_{}_time_{}_batch_{}.pkl'.format(length, time, batch),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


def diffusion_coefficient_net_val_data():
    x_val = []
    y_val = []
    for i in range(math.ceil(4800 / batch)):
        out, label = generate_batch_diffusion_coefficient_net(batch,model_range,length, time)
        x_val.append(out)
        y_val.append(label)
    with open('networks/val_data/diffusion_net/x_val_len_{}_time_{}_batch_{}_range_{}.pkl'.format(length, time, batch,model_range),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/diffusion_net/y_val_len_{}_time_{}_batch_{}_range_{}.pkl'.format(length, time, batch,model_range),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


def hurst_net_val_data():
    x_val = []
    y_val = []
    for i in range(math.ceil(8000 / batch)):
        out, label = generate_batch_hurst_net(batch, length, time)
        x_val.append(out)
        y_val.append(label)
    with open('networks/val_data/hurst_net/x_val_len_{}_time_{}_batch_{}_fbm_type_{}.pkl'.format(length, time, batch,fbm_type),
              'wb') as x_val_data:
        pickle.dump(x_val, x_val_data)
    with open('networks/val_data/hurst_net/y_val_len_{}_time_{}_batch_{}_fbm_type_{}.pkl'.format(length, time, batch,fbm_type),
              'wb') as y_val_data:
        pickle.dump(y_val, y_val_data)


if __name__ == '__main__':
    length = 50
    time = 1.0
    batch = 32
    model_range = '2-State-OD'
    fbm_type = 'Subdiffusive'
    classification_net_val_data()
