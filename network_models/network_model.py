from mongoengine import ListField, DictField, StringField, Document, IntField
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


class NetworkModel(Document):
    track_length = IntField(required=True)
    history = ListField(required=False)
    model_params = DictField(required=False)
    model_file = StringField(required=True)
    keras_model = None

    meta = {'allow_inheritance': True}

    def train_network(self, batch_size, track_time):
        pass

    def evaluate_track_input(self, track):
        pass

    def load_model_from_file(self):
        try:
            self.keras_model = load_model(self.model_file, compile=True)
        except ValueError:
            print("File does not exist!")

    def plot_loss_model(self, train=True, val=True):
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if train:
            plt.plot(np.arange(1, 11, 1), self.history['loss'], label="Train loss")
        if val:
            plt.plot(np.arange(1, 11, 1), self.history['val_loss'], label="Val loss")
        plt.legend()
        plt.show()

    def plot_mse_model(self, train=True, val=True):
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if train:
            plt.plot(np.arange(1, 11, 1), self.history['mse'], label="Train loss")
        if val:
            plt.plot(np.arange(1, 11, 1), self.history['val_mse'], label="Val loss")
        plt.legend()
        plt.show()

    def plot_accuracy_model(self, train=True, val=True, categorical=False):
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if categorical:
            if train:
                plt.plot(np.arange(1, 11, 1), self.history['categorical_accuracy'], label="Train loss")
            if val:
                plt.plot(np.arange(1, 11, 1), self.history['val_categorical_accuracy'], label="Val loss")
        else:
            if train:
                plt.plot(np.arange(1, 11, 1), self.history['acc'], label="Train loss")
            if val:
                plt.plot(np.arange(1, 11, 1), self.history['val_acc'], label="Val loss")

        plt.legend()
        plt.show()
