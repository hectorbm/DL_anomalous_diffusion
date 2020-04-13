from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix_for_layer(model, layer_name, ground_truth, predicted_value, labels, normalized):
    # Create df with confusion matrix
    confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=predicted_value)
    if normalized:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
    sns.set(font_scale=1)
    color_map = sns.color_palette(palette="Blues", n_colors=7)
    sns.heatmap(data=df_cm, annot=True, annot_kws={"size": 12}, cmap=color_map)
    # Plot matrix
    plt.title(f'Confusion Matrix {layer_name}')
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.show()


def plot_loss_model(history, train=True, val=True):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if train:
        plt.plot(np.arange(1, 11, 1), history['loss'], label="Train loss")
    if val:
        plt.plot(np.arange(1, 11, 1), history['val_loss'], label="Val loss")
    plt.legend()
    plt.show()


def plot_mse_model(history, train=True, val=True):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if train:
        plt.plot(np.arange(1, 11, 1), history['mse'], label="Train loss")
    if val:
        plt.plot(np.arange(1, 11, 1), history['val_mse'], label="Val loss")
    plt.legend()
    plt.show()


def plot_accuracy_model(history, train=True, val=True, categorical=False):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if categorical:
        if train:
            plt.plot(np.arange(1, 11, 1), history['categorical_accuracy'], label="Train loss")
        if val:
            plt.plot(np.arange(1, 11, 1), history['val_categorical_accuracy'], label="Val loss")
    else:
        if train:
            plt.plot(np.arange(1, 11, 1), history['acc'], label="Train loss")
        if val:
            plt.plot(np.arange(1, 11, 1), history['val_acc'], label="Val loss")

    plt.legend()
    plt.show()

