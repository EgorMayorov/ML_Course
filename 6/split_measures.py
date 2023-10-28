import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    freq = {}
    for val in sample:
        freq[val] = freq.get(val, 0) + 1

    size = len(sample)
    gini = 0.
    entropy = 0.
    freq_max = 0.
    for val in freq.keys():
        freq[val] = float(freq[val] / size)
        freq_max = max(freq_max, freq[val])

        entropy -= freq[val] * np.log(freq[val])

        gini -= freq[val] ** 2

    measures = {'gini': float(1. + gini), 'entropy': float(entropy), 'error': float(1. - freq_max)}
    return measures
