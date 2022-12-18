from matplotlib import pyplot as plt
import numpy as np

def compute_accuracy(targets, predictions):
    return (predictions == targets).mean()

def compute_confusion_matrix(targets, predictions, n_classes):

    confusion_matrix = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            confusion_matrix[i,j] = np.sum(np.logical_and(targets == i, predictions == j).astype(int))
    
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, labels, title):

    cmap = plt.cm.Blues
    ticks = np.arange(len(labels))

    plt.figure(figsize=(8,8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')