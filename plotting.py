"""Scripts for generating plots."""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import utils


def spike_hist():
    """Generates a histogram of spike counts over all the data."""

    plt.figure()
    for i, (_, spikes) in enumerate(utils.get_data_set('train')):
        print(i)

        x = np.reshape(spikes, (-1,))
        x = x[np.isnan(x) == False]

        plt.subplot(5, 2, i + 1)
        plt.hist(x, range(6), log=True)

    plt.show()


def calcium_hist():
    """Generates a histogram of calcium fluorescences over all the data."""

    plt.figure()
    for i, (calcium, _) in enumerate(utils.get_data_set('train')):
        print(i)

        x = np.reshape(calcium, (-1,))
        x = x[np.isnan(x) == False]

        plt.subplot(5, 2, i + 1)
        plt.hist(x, log=True)

    plt.show()


if __name__ == '__main__':
    # spike_hist()
    calcium_hist()
