from mongoengine import DictField, StringField, Document, IntField, FloatField, FileField, BooleanField
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.models import load_model


class NetworkModel(Document):
    track_length = IntField(required=True)
    track_time = FloatField(required=True)
    history = DictField(required=False)
    model_filename = StringField(required=False)
    model_file = FileField(required=False)
    hiperparams_opt = BooleanField(default=False)
    keras_model = None

    # Net params define optimal setup. Analysis params is for optimization
    net_params = DictField(required=False)
    analysis_params = {}

    meta = {'allow_inheritance': True}

    def __init__(self, *args, **values):
        super().__init__(*args, **values)
        if self.id is None:
            self.save()
        self.model_filename = ''.join(['models/', str(self.id), '.h5'])

    def train_network(self, batch_size):
        pass

    def evaluate_track_input(self, track):
        pass

    def validate_test_data_accuracy(self, n_axes, normalized=True):
        pass

    def validate_test_data_mse(self, n_axes):
        pass

    def build_model(self):
        pass

    def convert_history_to_db_format(self, history_training):
        for k, v in history_training.history.items():
            for i in range(len(history_training.history[k])):
                history_training.history[k][i] = float(history_training.history[k][i])
        self.history = history_training.history

    def load_model_from_file(self, only_local_files=False):
        load_success = False
        try:
            self.keras_model = load_model(self.model_filename, compile=True)
            load_success = True
        except (ValueError, OSError) as e:
            if not only_local_files:
                try:
                    print("Local model not available, loading from db")
                    self.keras_model = self.build_model()
                    weights = pickle.loads(self.model_file.read())
                    if weights is not None and self.keras_model is not None:
                        self.keras_model.set_weights(weights)
                        load_success = True
                    else:
                        raise TypeError

                except TypeError:
                    print("Weights not available in DB")
        return load_success

    def save_model_file_to_db(self):
        self.model_file.put(pickle.dumps(self.keras_model.get_weights()))

    def plot_loss_model(self, train=True, val=True):
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        if train:
            plt.plot(np.arange(1, len(self.history['loss']) + 1, 1), self.history['loss'], label="Train loss")
        if val:
            plt.plot(np.arange(1, len(self.history['val_loss']) + 1, 1), self.history['val_loss'], label="Val loss")
        plt.legend()
        plt.show()

    def plot_mse_model(self, train=True, val=True):
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model MSE')
        if train:
            plt.plot(np.arange(1, len(self.history['mse']) + 1, 1), self.history['mse'], label="Train mse")
        if val:
            plt.plot(np.arange(1, len(self.history['val_mse']) + 1, 1), self.history['val_mse'], label="Val mse")
        plt.legend()
        plt.show()

    def plot_accuracy_model(self, train=True, val=True, categorical=False):
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Accuracy')
        if categorical:
            if train:
                plt.plot(np.arange(1, len(self.history['categorical_accuracy']) + 1, 1),
                         self.history['categorical_accuracy'], label="Train categorical accuracy")
            if val:
                plt.plot(np.arange(1, len(self.history['val_categorical_accuracy']) + 1, 1),
                         self.history['val_categorical_accuracy'], label="Val categorical accuracy")
        else:
            if train:
                plt.plot(np.arange(1, len(self.history['acc']) + 1, 1), self.history['acc'],
                         label="Train accuracy")
            if val:
                plt.plot(np.arange(1, len(self.history['val_acc']) + 1, 1), self.history['val_acc'],
                         label="Val accuracy")

        plt.legend()
        plt.show()

    def is_valid_network_track_time(self, track_time):
        if (self.track_time * 0.85) <= track_time <= (self.track_time * 1.15):
            return True
        else:
            return False

