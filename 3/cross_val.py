import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    arr = np.arange(num_objects)
    r = (num_objects // num_folds) * (num_folds - 1)
    arr1 = arr[:r]
    arr2 = arr[r:]
    arr1 = arr1.reshape(-1, num_objects // num_folds)
    arr2 = arr2.reshape(-1, num_objects // num_folds + num_objects % num_folds)
    lst = []
    for i in range(arr1.shape[0]):
        tup = (np.concatenate((arr1[:i, :], arr1[i + 1:, :], arr2), axis=None), arr1[i, :])
        lst.append(tup)
    tup = (np.concatenate((arr2[:0, :], arr2[1:, :], arr1), axis=None), arr2[0, :])
    lst.append(tup)
    return lst
    # pass


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    cv = folds
    if cv is not None:
        n_folds = len(cv)
    else:
        n_folds = 3
        cv = kfold_split(len(X), n_folds)
    first, second = zip(*cv)
    lst1 = []
    lst2 = []
    for i in parameters['n_neighbors']:
        for j in parameters['normalizers']:
            for k in parameters['metrics']:
                for p in parameters['weights']:
                    lst1.append((j[1], i, k, p))
                    lst2.append(j[0])
    d = dict.fromkeys(lst1)
    d1 = dict.fromkeys(lst1)
    for i in lst1:
        d[i] = np.empty(n_folds)
    jj = -1
    for j in lst1:
        jj += 1
        KNN = knn_class(n_neighbors=j[1], weights=j[3], metric=j[2])
        for i in range(n_folds):
            if (j[0] != 'None'):
                a = lst2[jj]
                a.fit(X[first[i]])
                X_new = a.transform(X)
                KNN.fit(X_new[first[i]], y[first[i]])
                predict = KNN.predict(X_new[second[i]])
                d[j][i] = score_function(y[second[i]], predict)
            else:
                KNN.fit(X[first[i]], y[first[i]])
                predict = KNN.predict(X[second[i]])
                d[j][i] = score_function(y[second[i]], predict)
        d1[j] = np.mean(d[j])
    return d1
    # pass
