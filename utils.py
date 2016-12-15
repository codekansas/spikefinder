"""Spikefinder utils."""

from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np


_DOWNLOAD_URL = 'http://spikefinder.codeneuro.org/'


def get_data_set(data_index, name='train'):
    """Iteratively loads datasets as Numpy arrays.

    Args:
        data_index: int, the index of the dataset to load.
        name: one of ['train', 'test'], the training set to load.

    Raises:
        ValueError: Invalid value for "name" provided.
    """

    if not 'DATA_PATH' in os.environ:
        raise ValueError('The environment variable "DATA_PATH" is not set. '
                         'It should be set to point to the directory where '
                         'the training and test data is located.')

    if name == 'train':
        data_path = os.path.join(os.environ['DATA_PATH'], 'spikefinder.train')
        file_names = ['%d.train.calcium.csv' % data_index,
                      '%d.train.spikes.csv' % data_index]
    elif name == 'test':
        data_path = os.path.join(os.environ['DATA_PATH'], 'spikefinder.test')
        file_names = ['%d.test.calcium.csv' % data_index]
    else:
        raise ValueError('Invalid data set: %s (should be either "train" or '
                         '"test")' % name)

    if not os.path.exists(data_path):
        raise ValueError('The training data was not found at %s. You '
                         'should download it from %s and extract the '
                         'zipped files to %s' % (data_path,
                                                 _DOWNLOAD_URL,
                                                 os.environ['DATA_PATH']))

    loaded_datasets = list()

    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_path):
            raise ValueError('File not found: %s' % file_path)
        loaded_datasets.append(np.genfromtxt(file_path,
                                             delimiter=',',
                                             skip_header=1))

    return loaded_datasets


def plot_dataset(dataset, *args, **kwargs):
    """Plots a dataset using Matplotlib.

    Since all the datasets were sampled at 100 Hz, the X-axis is seconds,
    with 100 samples per second.

    Args:
        dataset: 2D Numpy array with shape (time_steps, channels) with the data
            to plot, or a 1D array with shape (time_steps).
        *args: extra arguments to plt.plot
        **kwargs: extra arguments to plt.plot
    """

    if dataset.ndim == 1:
        time_steps, = dataset.shape
    elif dataset.ndim == 2:
        time_steps, _ = dataset.shape
    else:
        raise ValueError('Invalid numver of dimensions: %d' % dataset.ndim)

    x = np.arange(time_steps) / 100.
    plt.plot(x, dataset, *args, **kwargs)


if __name__ == '__main__':
    calcium_data, spikes_data = get_data_set(1, name='train')
    plot_dataset(calcium_data[:, 0])
    plot_dataset(spikes_data[:, 0], color='k')
    plt.show()
