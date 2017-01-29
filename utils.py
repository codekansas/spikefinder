"""Spikefinder utils for loading and visualizing the data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import random
import os

import numpy as np

from keras.layers import Layer

import keras.backend as K
import tensorflow as tf


_DOWNLOAD_URL = 'http://spikefinder.codeneuro.org/'


class DeltaFeature(Layer):
    """Layer for calculating time-wise deltas."""

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('DeltaFeature input should have three '
                             'dimensions. Got %d.' % len(input_shape))
        super(DeltaFeature, self).build(input_shape)

    def call(self, x, mask=None):
        return K.concatenate([x[:1], x[:-1]], 0)

    def get_output_shape_for(self, input_shape):
        return input_shape


class QuadFeature(Layer):
    """Layer for calculating quadratic feature (square inputs)."""

    def call(self, x, mask=None):
        return K.square(x)

    def get_output_shape_for(self, input_shape):
        return input_shape


def pad_to_length(x, length, axis=0):
    """Pads `x` to length `length` along axis `axis`."""

    s = list(x.shape)
    s[axis] = length
    z = np.zeros(s)
    z[:x.shape[axis]] = x
    return z


def output_to_ints(spike_output):
    """Converts spikes from range to integer values."""

    return np.squeeze(np.floor(spike_output))


def pearson_corr(y_true, y_pred):
    """Calculates Pearson correlation as a metric.

    This calculates Pearson correlation the way that the competition calculates
    it (as integer values).

    y_true and y_pred have shape (batch_size, num_timesteps, 1).
    """

    y_true = K.squeeze(tf.floor(y_true), 2)
    y_pred = K.squeeze(tf.floor(y_pred), 2)

    x_mean = y_true - K.mean(y_true, axis=1, keepdims=True)
    y_mean = y_pred - K.mean(y_pred, axis=1, keepdims=True)

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean, axis=1)
    d = (K.sum(K.square(x_mean), axis=1) *
         K.sum(K.square(y_mean), axis=1))

    return K.mean(n / (K.sqrt(d) + 1e-12))


def pearson_loss(y_true, y_pred):
    """Loss function to maximize pearson correlation.

    y_true and y_pred have shape (batch_size, num_timesteps, 1).
    """

    # Removes the last dimension.
    y_true = K.squeeze(y_true, 2)
    y_pred = K.squeeze(y_pred, 2)

    x_mean = y_true - K.mean(y_true, axis=1, keepdims=True)
    y_mean = y_pred - K.mean(y_pred, axis=1, keepdims=True)

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean, axis=1)
    d = (K.sum(K.square(x_mean), axis=1) *
         K.sum(K.square(y_mean), axis=1))

    # Maximize corr by minimizing negative.
    corr = n / (K.sqrt(d + 1e-12))

    # Add a bit of MSE loss, to put stuff in the right place.
    # loss = K.mean(K.square(y_pred - y_true), axis=-1) * 0.1

    return -corr


def bin_percent(i):
    """Metric that keeps track of percentage of outputs in each bin."""

    def _prct(y_true, y_pred):
        y_true = tf.floor(y_true)
        y_pred = tf.floor(y_pred)

        return {
            '%d' % i: K.mean(K.equal(y_pred, i)),
            # '%d_true' % i: K.mean(K.equal(y_true, i)),
        }

    return _prct


def get_eval(dataset, num_timesteps=100):
    """Iterates through columns of the dataset.

    Args:
        dataset: str, "train" or "test".
        num_timesteps: int, number of timesteps in each batch.

    Yields:
        Consult code (multiple layers).
    """

    def _process_single_column(calcium_column):
        calcium_column = np.expand_dims(calcium_column, -1)
        col_length = len(calcium_column) - np.sum(np.isnan(calcium_column))

        # Removes the NaN values.
        calcium_column = calcium_column[:col_length]

        arr_list = []
        for i in range(0, col_length, num_timesteps):
            arr_list.append(pad_to_length(calcium_column[i:i + num_timesteps]))

        return col_length, np.stack(arr_list)

    def _entry_iterator(data_entry):
        for col_idx in range(data_entry.shape[1]):
            col_length, data = _process_single_column(data_entry[:, col_idx])
            yield col_idx, col_length, data

    if dataset == 'train':
        for d_idx, (data_entry, _) in enumerate(get_data_set('train')):
            yield (d_idx, data_entry.shape, _entry_iterator(data_entry))

    elif dataset == 'test':
        for d_idx, data_entry in enumerate(get_data_set('test')):
            yield (d_idx, data_entry.shape, _entry_iterator(data_entry))

    else:
        raise ValueError('Invalid dataset: "%s" (expected "train" or '
                         '"test").' % dataset)

def get_training_set(num_timesteps=100, cache='/tmp/spikefinder_data.npz'):
    """Builds the training set (as Numpy arrays).

    Args:
        num_timesteps: int, number of timesteps in each batch.
        cache: str, where to cache the built dataset.

    Returns:
        tuple, (dataset, calcium, spikes)
            dataset: Numpy integer array, the dataset in question.
            calcium: Numpy float array, the calcium traces.
            spikes: Numpy float array, the one-hot encoded spikes.
    """

    if not os.path.exists(cache):

        def _process_single_column(calcium_column, spikes_column):
            calcium_column = np.expand_dims(calcium_column, -1)
            spikes_column = np.expand_dims(spikes_column, -1)
            col_length = len(calcium_column) - np.sum(np.isnan(calcium_column))

            # Removes the NaN values.
            calcium_column = calcium_column[:col_length]
            spikes_column = spikes_column[:col_length]

            for i in range(num_timesteps, col_length, num_timesteps):
                yield (pad_to_length(calcium_column[i:i + num_timesteps],
                                     num_timesteps),
                       pad_to_length(spikes_column[i:i + num_timesteps],
                                     num_timesteps))

        pairs = ([_process_single_column(c[:, i], s[:, i])
                  for i in range(c.shape[1])]
                 for c, s in get_data_set('train'))

        # Builds actual arrays.
        dataset_arr = []
        calcium_arr = []
        spikes_arr = []
        for dataset, pair in enumerate(pairs):
            dataset = np.asarray([dataset])
            for calcium, spikes in itertools.chain.from_iterable(pair):
                dataset_arr.append(dataset)
                calcium_arr.append(calcium)
                spikes_arr.append(spikes)

        # Concatenates to one.
        dataset_arr = np.stack(dataset_arr)
        calcium_arr = np.stack(calcium_arr)
        spikes_arr = np.stack(spikes_arr)

        with open(cache, 'wb') as f:
            np.savez(f,
                     dataset=dataset_arr,
                     calcium=calcium_arr,
                     spikes=spikes_arr)

    with open(cache, 'rb') as f:
        npzfile = np.load(f)
        dataset = npzfile['dataset']
        calcium = npzfile['calcium']
        spikes = npzfile['spikes']

    if calcium.shape[1] != num_timesteps or spikes.shape[1] != num_timesteps:
        raise ValueError('Old cached files were found at "%s". Delete these, '
                         'then re-run.' % cache)

    return dataset, calcium, spikes


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
