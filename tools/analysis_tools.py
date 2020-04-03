from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix_for_layer(model, layer,ground_trouth, predicted_value, labels,normalized):
    #Create df with confusion matrix
    confusion_mat = confusion_matrix(ground_trouth,predicted_value)
    if normalized:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    print(confusion_mat)
    df_cm = pd.DataFrame(confusion_mat,labels,labels)
    sns.set(font_scale=1)
    cmap = sns.color_palette("Blues", 7)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 10},cmap=cmap)
    #Plot matrix
    plt.title(f'Confusion Matrix Layer for {layer}')
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.show()