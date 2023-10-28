import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        # pass

    def transform(self, data=None):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        if data is None:
            return (self - np.min(self, axis=0)) / (
                    np.max(self, axis=0) - np.min(self, axis=0))
        return (data - self.min) / (self.max - self.min)
        # pass


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.d = np.std(data, axis=0)
        self.e = np.mean(data, axis=0)
        # pass

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.e) / self.d
        # pass
