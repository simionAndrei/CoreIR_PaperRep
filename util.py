import numpy as np

def label_based_accuracy(y_true, y_pred):

    accuracies = []
    for i in range(y_true.shape[0]):
        y_true_idx = set(y_true[i].nonzero()[0])
        y_pred_idx = set(y_pred[i].nonzero()[0])

        num_correct_labels = len(y_pred_idx.intersection(y_true_idx))
        num_union_labels   = len(y_pred_idx.union(y_true_idx))

        accuracies.append(num_correct_labels/ num_union_labels)

    return np.mean(accuracies)