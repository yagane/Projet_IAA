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

    cmap = plt.cm.YlOrBr
    ticks = np.arange(len(labels))
    thresh = confusion_matrix.max()/2. # Threshold for text printing (black or white)

    plt.figure(figsize=(8,8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    
    for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                    plt.text(j, i, "%i"%(confusion_matrix[i, j]),
                                ha="center", va="center",
                                color='white' if confusion_matrix[i, j] > thresh else 'black')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')