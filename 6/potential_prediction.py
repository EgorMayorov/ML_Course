import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        x_transformed = []
        for potential in x:
            potential_pad = potential
            if len(potential_pad[potential_pad == np.min(potential_pad)]) == 1:
                potential_pad = np.pad(potential, 128, constant_values=20)
                center_idx = np.unravel_index(np.argmin(potential_pad), potential_pad.shape)
                potential_centered = np.roll(potential_pad, np.array(potential_pad.shape) // 2 - center_idx, axis=(0, 1))
                M, N = potential_centered.shape
                K = 2
                L = 2
                MK = M // K
                NL = N // L
                potential_minpool = potential_centered[:MK * K, :NL * L].reshape(MK, K, NL, L).min(axis=(1, 3))
                x_transformed.append(potential_minpool)
            else:
                idx_0, idx_1 = np.where(potential_pad != 20)
                min_idx_0 = min(idx_0)
                max_idx_0 = max(idx_0)
                min_idx_1 = min(idx_1)
                max_idx_1 = max(idx_1)
                center_idx = ((max_idx_0 - min_idx_0) // 2 + min_idx_0, (max_idx_1-min_idx_1) // 2 + min_idx_1)
                potential_centered = np.roll(potential_pad, np.array(potential_pad.shape) // 2 - center_idx, axis=(0, 1))
                x_transformed.append(potential_centered)
        return np.stack(x_transformed, axis=0).reshape((x.shape[0], -1))


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    model = Pipeline([('transformer', PotentialTransformer()), ('forest', ExtraTreesRegressor(n_estimators=1000,
                                                                                              criterion='friedman_mse',
                                                                                              max_features=0.0009,
                                                                                              random_state=3,
                                                                                              n_jobs=-1))])
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
