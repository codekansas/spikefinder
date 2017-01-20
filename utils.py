"""Spikefinder utils for loading and visualizing the data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import random
import os

import numpy as np


_DOWNLOAD_URL = 'http://spikefinder.codeneuro.org/'


def generate_training_set(num_timesteps=100, batch_size=32):
    """Generates the training dataset.

    When choosing num_timesteps, keep in mind that a sampling rate of 100 Hz
    was used to collect the data.

    Args:
        num_timesteps: int, number of timesteps before and after to generate.
        batch_size: int, number of samples per batch.

    Yields:
        tuples (calcium, spikes, last_spike) where:
            calcium: Numpy array with shape (num_timesteps, 1), where each
                value represents the calcium recording at that timestep.
            spikes: Numpy array with shape (num_timesteps - 1, 1), where each
                value represents the number of spikes in that interval.
            last_spike: Numpy array with shape (1), the number of spikes in
                the last interval (the model should try to predict this).
    """

    def _process_single_column(calcium_column, spikes_column):
        calcium_column = np.expand_dims(calcium_column, -1)
        spikes_column = np.cast['int32'](spikes_column)
        column_length = len(calcium_column) - np.sum(np.isnan(calcium_column))

        while True:  # Iterates infinitely.
            idx = range(num_timesteps, column_length, num_timesteps)
            random.shuffle(idx)
            for i in idx:
                yield (calcium_column[i - num_timesteps:i],
                       spikes_column[i - num_timesteps:i])

    # Iterating this way avoids caching the results (i.e. itertools.repeat)
    pairs = [[_process_single_column(calcium[:, i], spikes[:, i])
              for i in range(calcium.shape[1])]
             for calcium, spikes in get_data_set('train')]

    eye = np.eye(7)
    while True:
        i = np.random.randint(0, len(pairs))
        j = np.random.randint(0, len(pairs[i]))
        dataset = pairs[i]
        calcium, spikes = dataset[j].next()
        yield i, calcium, eye[spikes]


def get_data_set(mode='train'):
    """Loads datasets as Numpy arrays.

    Args:
        mode: one of ['train', 'test'], the training set to load.

    Yields:
        Lists of Numpy arrays representing the loaded dataset.

    Raises:
        ValueError: Invalid value for "mode" provided.
    """

    if not 'DATA_PATH' in os.environ:
        raise ValueError('The environment variable "DATA_PATH" is not set. '
                         'It should be set to point to the directory where '
                         'the training and test data is located.')

    if mode == 'train':
        data_path = os.path.join(os.environ['DATA_PATH'], 'spikefinder.train')
        file_names = [('%d.train.calcium.csv' % i, '%d.train.spikes.csv' % i)
                      for i in range(1, 11)]
    elif mode == 'test':
        data_path = os.path.join(os.environ['DATA_PATH'], 'spikefinder.test')
        file_names = [('%d.test.calcium.csv' % i,) for i in range(1, 6)]
    else:
        raise ValueError('Invalid mode: %s (should be either "train" or '
                         '"test")' % mode)

    if not os.path.exists(data_path):
        raise ValueError('The training data was not found at %s. You '
                         'should download it from %s and extract the '
                         'zipped files to %s' % (data_path,
                                                 _DOWNLOAD_URL,
                                                 os.environ['DATA_PATH']))

    for train_or_test_set in file_names:
        loaded_dataset = []

        for file_name in train_or_test_set:
            file_path = os.path.join(data_path, file_name)
            if not os.path.exists(file_path):
                raise ValueError('File not found: %s' % file_path)
            loaded_dataset.append(np.genfromtxt(file_path,
                                                delimiter=',',
                                                skip_header=1))

        yield loaded_dataset


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

    # The import is done here because of an issue with matplotlib and MacOS.
    import matplotlib.pyplot as plt

    if dataset.ndim == 1:
        time_steps, = dataset.shape
    elif dataset.ndim == 2:
        time_steps, _ = dataset.shape
    else:
        raise ValueError('Invalid number of dimensions: %d' % dataset.ndim)

    x = np.arange(time_steps) / 100.
    plt.plot(x, dataset, *args, **kwargs)
