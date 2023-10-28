import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    """
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    """
    labs = np.unique(labels, return_counts=True)[0]
    sizes = np.unique(labels, return_counts=True)[1]
    if len(labs) == 1:
        return 0
    distances = sklearn.metrics.pairwise_distances(x, force_all_finite=True)
    d = np.zeros((x.shape[0], 1))
    s = np.zeros((x.shape[0], 1))
    for ll, size in zip(labs, sizes):
        if size == 1:
            s[labels == ll] = 0
            d[labels == ll] = 0
        else:
            s[labels == ll] = (distances[labels == ll][:, labels == ll] / (size - 1)).sum(axis=1).reshape((-1, 1))
            matrix = np.hstack(
                [(distances[labels == ll][:, labels == c] / sizes[np.where(labs == c)]).sum(axis=1).reshape((-1, 1)) for c in
                 labs if c != ll])
            d[labels == ll] = np.min(matrix, axis=1).reshape((-1, 1))
    s_d_max = np.max(np.hstack([d, s]), axis=1).reshape((-1, 1))
    return np.mean(np.where(s_d_max == 0, 0, np.divide(d - s, s_d_max, where=s_d_max != 0)))


def bcubed_score(true_labels, predicted_labels):
    """"
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    """
    L = true_labels[:, np.newaxis] == true_labels[np.newaxis, :]
    C = predicted_labels[:, np.newaxis] == predicted_labels[np.newaxis, :]
    prec = np.mean(np.sum(L * C, axis=1) / C.sum(axis=1))
    recall = np.mean(np.sum(L * C, axis=1) / L.sum(axis=1))
    return 2 * (prec * recall) / (prec + recall)
