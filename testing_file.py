from networks.diffusion_coeff_network_model import DiffusionCoefficientNetworkModel
from networks.l1_network_model import L1NetworkModel
from networks.l2_network_model import L2NetworkModel
from networks.state_detection_network_model import StateDetectionNetworkModel
from tools.db_connection import *
from networks.network_model import NetworkModel
from networks.noise_reduction_network_model import NoiseReductionNetworkModel
from tracks.file import File
from tracks.tracks import Tracks
import matplotlib.pyplot as plt

"""
if __name__ == '__main__':
    filename = 'test_data/1-mAb35_CDx_0min.csv'
    file = File()
    connect_to_db()
    file.add_raw_file(filename=filename)
    file.save()
    tracks = file.load_local_file(filename=filename)
    for track in tracks:
        track.save()
        if track.track_length > 45:
            track.plot_xy()
            track.plot_axis_velocity(1)
            track.plot_axis_velocity(2)

            track.plot_axis_with_time(1)
            track.plot_axis_with_time(2)
    disconnect_to_db()

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

"""
if __name__ == '__main__':
    connect_to_db()
    model = NoiseReductionNetworkModel(track_length=15, diffusion_model_state=1)
    #model = NoiseReductionNetworkModel.objects(id='5e97860926231d1674b58efd')[0]
    # noise_reduction_model = NoiseReductionNetworkModel.objects(id='')[0]
    # noise_reduction_model.save()
    # model.train_network(batch_size=64, track_time=0.2)
    # model.save()
    # model.plot_mse_model()
    # model.plot_loss_model()
    for track in Tracks.objects():
        if track.track_length == model.track_length:
            X = model.evaluate_track_input(track)
    disconnect_to_db()

"""

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
if __name__ == '__main__':
    connect_to_db()
    model = DiffusionCoefficientNetworkModel(track_length=15, diffusion_model_state=0)
    model.train_network(batch_size=32, track_time=0.2)
    model.save()
    for track in Tracks.objects():
        if track.track_length == model.track_length:
            print(model.evaluate_track_input(track))
    disconnect_to_db()

"""
