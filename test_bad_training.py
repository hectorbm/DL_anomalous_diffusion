from Networks.diffusion_coefficient_network import DiffusionCoefficientNetworkModel
from Tools.db_connection import disconnect_to_db, connect_to_db
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    connect_to_db()
    lower_limit = 15
    upper_limit = 1500
    nets = DiffusionCoefficientNetworkModel.objects(track_length__in=range(lower_limit, upper_limit), hiperparams_opt=False)
    count = 0
    for net in nets:
        print(net.id)
        epochs = np.arange(1, len(net.history['mse'])+1)
        plt.plot(epochs, net.history['mse'])
        if net.history['mse'][-1] >= 0.006:
            count += 1
    print('Must be retrained:{}'.format(count))

    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()

    for net in nets:
        if net.history['mse'][-1] >= 0.006:
            net.net_params['lr'] = 6e-8
            # net.net_params['lr'] = 5e-7

            net.train_network()
            net.load_model_from_file()
            net.update_model_file_to_db()
            net.save()

    disconnect_to_db()
