from tools.db_connection import *
from network_models.network_model import NetworkModel
"""
if __name__ == '__main__':
    filename = 'test_data/1-mAb35_CDx_0min.csv'
    file = File()
    connect_to_db()
    file.add_raw_file(filename=filename)
    file.save()
    tracks = file.load_local_file(filename=filename)
    for track in tracks:
        #track.save()
        if track.track_length > 45:
            track.plot_xy()
    disconnect_to_db()

"""

if __name__ == '__main__':
    connect_to_db()
    model = NetworkModel.objects()[0]
    #model.model_file = 'models/first_layer_1.h5'
    #model.save()
    model.load_model_from_file()
    model.keras_model.summary()
    disconnect_to_db()
