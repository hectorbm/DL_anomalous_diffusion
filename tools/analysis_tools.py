from scipy.optimize import curve_fit
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix_for_layer(layer_name, ground_truth, predicted_value, labels, normalized):
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


def linear_func(x, beta, d):
    return d * (x ** 1)


def mean_squared_displacement(x, y, time_length, non_linear=True):
    data = np.sqrt(x ** 2 + y ** 2)
    n_data = np.size(data)
    number_of_delta_t = np.int((n_data - 1))
    t_vec = np.arange(1, np.int(number_of_delta_t))

    msd = np.zeros([len(t_vec), 1])
    for dt, ind in zip(t_vec, range(len(t_vec))):
        squared_displacement = (data[1 + dt:] - data[:-1 - dt]) ** 2
        msd[ind] = np.mean(squared_displacement, axis=0)

    msd = np.array(msd)

    t_vec = np.linspace(0.0001, time_length, len(x)-2)
    msd = np.array(msd).ravel()
    if non_linear:
        a, b = curve_fit(linear_func, t_vec, msd, bounds=((0, 0), (2, np.inf)), maxfev=2000)
    else:
        a, b = curve_fit(linear_func, t_vec, msd, bounds=((0, 0, -np.inf), (2, np.inf, np.inf)), maxfev=2000)

    return t_vec, msd, a
