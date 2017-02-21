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

if K.backend() == 'tensorflow':
    from tensorflow import floor
else:
    from theano.tensor import floor


_DOWNLOAD_URL = 'http://spikefinder.codeneuro.org/'


class DeltaFeature(Layer):
    """Layer for calculating time-wise deltas."""

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('DeltaFeature input should have three '
                             'dimensions. Got %d.' % len(input_shape))
        super(DeltaFeature, self).build(input_shape)

    def call(self, x, mask=None):
        x_a, x_b = K.zeros_like(x[:, 1:]), x[:, :1]
        x_shifted = K.concatenate([x_a, x_b], axis=1)
        return x - x_shifted

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


def _normalize(i):
    min_v = K.min(i)
    max_v = K.max(i)
    return (i - min_v) * 2 / (max_v - min_v + 1e-7)


def pearson_corr(y_true, y_pred, pre_floor=False, normalize=False):
    """Calculates Pearson correlation as a metric.

    This calculates Pearson correlation the way that the competition calculates
    it (as integer values).

    y_true and y_pred have shape (batch_size, num_timesteps, 1).
    """

    if normalize:
        y_pred = _normalize(y_pred)

    if pre_floor:
        y_true = K.squeeze(floor(y_true), 2)
        y_pred = K.squeeze(floor(y_pred), 2)
    else:
        y_true = K.squeeze(y_true, 2)
        y_pred = K.squeeze(y_pred, 2)

    x_mean = y_true - K.mean(y_true, axis=1, keepdims=True)
    y_mean = y_pred - K.mean(y_pred, axis=1, keepdims=True)

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean, axis=1)
    d = (K.sum(K.square(x_mean), axis=1) *
         K.sum(K.square(y_mean), axis=1))

    return K.mean(n / (K.sqrt(d) + 1e-12))


def pearson_loss(y_true, y_pred, depth=1, normalize=False):
    """Loss function to maximize pearson correlation.

    y_true and y_pred have shape (batch_size, num_timesteps, 1).
    """

    if normalize:
        y_pred = _normalize(y_pred)

    x_mean = y_true - K.mean(y_true, axis=1, keepdims=True)
    y_mean = y_pred - K.mean(y_pred, axis=1, keepdims=True)

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean, axis=1)
    d = (K.sum(K.square(x_mean), axis=1) *
         K.sum(K.square(y_mean), axis=1))

    # Maximize corr by minimizing negative.
    corr = n / (K.sqrt(d + 1e-12))
    loss = -corr

    # Add a bit of MSE loss, to put stuff in the right place.
    # loss = K.mean(K.square(y_pred - y_true), axis=-1) * 0.1

    if depth > 0:
        _pool = lambda x: x[:, 1:] + x[:, :-1]
        loss = loss + 2 * pearson_loss(
                y_true=_pool(y_true),
                y_pred=_pool(y_pred),
                depth=depth - 1,
                normalize=False)

    return loss


def stats(_, y_pred):
    """Metric that keeps track of some statistics."""

    return {
        'mean': K.mean(y_pred),
        # 'max': K.max(y_pred),
        # 'min': K.min(y_pred),
        'std': K.std(y_pred),
    }


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
            arr_list.append(pad_to_length(calcium_column[i:i + num_timesteps],
                                          num_timesteps))

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

def get_training_set(num_timesteps=100,
                     cache='/tmp/spikefinder_data.npz',
                     rebuild=False):
    """Builds the training set (as Numpy arrays).

    Args:
        num_timesteps: int, number of timesteps in each batch.
        cache: str, where to cache the built dataset.
        rebuild: bool, if set, ignore the cache.

    Returns:
        tuple, (dataset, calcium, spikes)
            dataset: Numpy integer array, the dataset in question.
            calcium: Numpy float array, the calcium traces.
            spikes: Numpy float array, the one-hot encoded spikes.
    """

    if not os.path.exists(cache) or rebuild:

        def _process_data_set(calcium, spikes):
            col_lens = calcium.shape[0] - np.sum(np.isnan(calcium), axis=0)
            calcium = np.expand_dims(calcium, -1)
            spikes = np.expand_dims(spikes, -1)

            # Converts NaNs to 0s.
            calcium_c = np.nan_to_num(calcium)
            spikes_c = np.nan_to_num(spikes)

            calcium_n = calcium / np.linalg.norm(calcium_c)

            step_size = num_timesteps  # // 3
            calcium_stats = np.concatenate(
                    [np.nanmean(calcium, axis=1),
                     np.nanstd(calcium, axis=1),
                     np.nanmedian(calcium, axis=1),
                     np.nanmean(calcium_n, axis=1),
                     np.nanstd(calcium_n, axis=1),
                     np.nanmedian(calcium_n, axis=1)],
                    axis=1)

            def _pad(x, i):
                return pad_to_length(x[i:i + num_timesteps], num_timesteps)

            for i in range(calcium.shape[1]):
                for j in range(0, col_lens[i] - step_size, step_size):
                    yield (_pad(calcium_c[:, i], j),
                           _pad(spikes_c[:, i], j),
                           _pad(calcium_stats, j))

        pairs = (_process_data_set(c, s) for c, s in get_data_set('train'))

        # Builds actual arrays.
        dataset_arr = []
        calcium_arr = []
        calcium_stats_arr = []
        spikes_arr = []
        for dataset, pair in enumerate(pairs):
            dataset = np.asarray([dataset])
            for c, s, c_stats in pair:
                dataset_arr.append(dataset)
                calcium_arr.append(c)
                calcium_stats_arr.append(c_stats)
                spikes_arr.append(s)
            print('processed %d datasets' % dataset)

        # Concatenates to one.
        dataset_arr = np.stack(dataset_arr)
        calcium_arr = np.stack(calcium_arr)
        calcium_stats_arr = np.stack(calcium_stats_arr)
        spikes_arr = np.stack(spikes_arr)

        # Shuffles along the batch axis.
        idx = np.arange(dataset_arr.shape[0])
        np.random.shuffle(idx)
        dataset_arr = dataset_arr[idx]
        calcium_arr = calcium_arr[idx]
        calcium_stats_arr = calcium_stats_arr[idx]
        spikes_arr = spikes_arr[idx]

        with open(cache, 'wb') as f:
            np.savez(f,
                     dataset=dataset_arr,
                     calcium=calcium_arr,
                     calcium_stats=calcium_stats_arr,
                     spikes=spikes_arr)

    with open(cache, 'rb') as f:
        npzfile = np.load(f)
        dataset = npzfile['dataset']
        calcium = npzfile['calcium']
        calcium_stats = npzfile['calcium_stats']
        spikes = npzfile['spikes']

    if calcium.shape[1] != num_timesteps:
        raise ValueError('Old cached files were found at "%s". Delete these, '
                         'then re-run.' % cache)

    return dataset, calcium, calcium_stats, spikes


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
