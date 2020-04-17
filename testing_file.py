import numpy

from networks.diffusion_coeff_network_model import DiffusionCoefficientNetworkModel
from networks.l1_network_model import L1NetworkModel
from networks.l2_network_model import L2NetworkModel
from networks.state_detection_network_model import StateDetectionNetworkModel
from tools.db_connection import *
from networks.network_model import NetworkModel
from networks.noise_reduction_network_model import NoiseReductionNetworkModel
from tracks.file import File, get_files_in_path
from tracks.tracks import Tracks
import matplotlib.pyplot as plt


def load_files_and_tracks(path_name):
    connect_to_db()
    files = get_files_in_path(path_name=path_name)
    for f in files:

        try:
            file = File()
            print('Processing file:{}'.format(f))
            file.parse_filename(filename=f)
            file.add_raw_file(filename=f)
            file.save()
            tracks = file.load_local_file(filename=f)
            for track in tracks:
                track.save()
        except AssertionError:
            print("Unable to load file:{}".format(f))
    disconnect_to_db()


"""
if __name__ == '__main__':
    path = 'experimental_data/btx/'
    load_files_and_tracks(path)

"""

"""
if __name__ == '__main__':
    connect_to_db()
    model = L1NetworkModel.objects(id='5e976ce557a6c42451718382')[0]
    model.validate_test_data_accuracy(n_axes=2, normalized=True)
    for track in Tracks.objects():
        if track.track_length == model.track_length:
            model.evaluate_track_input(track)

    disconnect_to_db()

"""

if __name__ == '__main__':
    connect_to_db()
    t = numpy.arange(0.05, 0.55, 0.022)
    print(len(t))
    for i in range(5, 25, 1):

        model = NoiseReductionNetworkModel(track_length=i, track_time=t[i], diffusion_model_state=1)
        # model = NoiseReductionNetworkModel.objects(id='5e98f269a4ad83dc5fbfec14')[0]
        # model.load_model_from_file()
        model.train_network(batch_size=64)
        model.save()

    #model.plot_mse_model()
    #mse_avg = model.validate_test_data_mse(n_axes=2)
    #print('MSE(Noise Reduction):{:.2}'.format(mse_avg))

    # model2 = DiffusionCoefficientNetworkModel(track_length=15, track_time=0.17, diffusion_model_state=1)
    # model2 = DiffusionCoefficientNetworkModel.objects(id='5e990d59f8cb2422eb7d2f9c')[0]
    # model2.load_model_from_file()
    # model2.save()
    # model2.set_noise_reduction_model(model)
    # model2.train_network(batch_size=32)
    # model2.save()
    # model2.plot_mse_model()
    # mse_avg2 = model2.validate_test_data_mse(n_axes=2)
    # print('MSE(Diffusion Coefficient):{:.2}'.format(mse_avg2))

    disconnect_to_db()

"""
if __name__ == '__main__':
    connect_to_db()
    model = L2NetworkModel.objects(id='5e977898537f5d994a6a880f')[0]
    #model.train_network(batch_size=32)
    #model.save()
    model.validate_test_data_accuracy(n_axes=2)
    for track in Tracks.objects():
        if track.track_length == model.track_length:
            model.evaluate_track_input(track)

    disconnect_to_db()

"""

"""
if __name__ == '__main__':
    connect_to_db()
    model = StateDetectionNetworkModel.objects(id='5e97822192a3e24e26f0dddb')[0]
    #model.train_network(batch_size=32, track_time=0.3)
    model.track_time = 0.2
    model.save()
    model.validate_test_data_accuracy(n_axes=2)
    tracks = list(Tracks.objects())
    for track in tracks:
        if track.track_length == model.track_length:
            model.evaluate_track_input(track)
    disconnect_to_db()

"""

"""


"""
