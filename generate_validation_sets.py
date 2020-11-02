import math
import pickle

from networks.generators import generate_batch_of_samples_l1


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


if __name__ == '__main__':
    length = 50
    time = 1.0
    batch = 32

    classification_net_val_data()
