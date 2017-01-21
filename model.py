"""Model definitions.

The model takes as input the calcium channel recordings and number of spikes
over some amount of time and tries to predict the number of spikes on the last
interval of the recording.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import utils

import keras.backend as K

from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import merge
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from keras.models import Model


def conv_bn(x, nb_filter, filter_length):
    """Applies convolution and batch normalization."""

    x = Convolution1D(nb_filter, filter_length,
                      activation='relu', border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)  # Normalizes across time.
    return x


def inception_cell(x):
    """Applies a single inception cell."""

    branch1x1 = conv_bn(x, 64, 1)

    branch5x5 = conv_bn(x, 48, 1)
    branch5x5 = conv_bn(branch5x5, 64, 5)

    branch3x3dbl = conv_bn(x, 64, 1)
    branch3x3dbl = conv_bn(branch3x3dbl, 96, 3)
    branch3x3dbl = conv_bn(branch3x3dbl, 96, 3)

    branch_pool = AveragePooling1D(3, stride=1, border_mode='same')(x)
    branch_pool = conv_bn(branch_pool, 32, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=-1)

    return x


def build_model(num_timesteps):
    dataset = Input(shape=(1,), dtype='int32', name='dataset')
    calcium = Input(shape=(num_timesteps, 1), dtype='float32', name='calcium')

    # Embed dataset into vector space.
    flat = Flatten()(RepeatVector(num_timesteps)(dataset))
    data_emb = Embedding(10, 1, init='orthogonal')(flat)

    # Normalizes the data along the time dimension.
    calcium_norm = BatchNormalization(mode=2, axis=1)(calcium)

    # Merge channels together.
    hidden = merge([calcium_norm, data_emb],
                   mode='concat', concat_axis=-1)

    # Adds convolutional layers.
    for _ in range(3):
        hidden = inception_cell(hidden)

    # Adds recurrent layers.
    hidden = Bidirectional(LSTM(64, return_sequences=True))(hidden)
    hidden = Bidirectional(LSTM(64, return_sequences=True))(hidden)

    # Adds output layer.
    output = TimeDistributed(Dense(7, activation='softmax',
                                   W_regularizer='l2'))(hidden)

    # Builds the model.
    model = Model(input=[dataset, calcium], output=[output])

    return model


def pearson_corr(y_true, y_pred):
    """Calculates Pearson correlation as a metric."""

    # Gets the argmax of each.
    y_pred = K.cast(K.argmax(y_pred, axis=-1), 'float32')
    y_true = K.cast(K.argmax(y_true, axis=-1), 'float32')

    x_mean = y_true - K.mean(y_true)
    y_mean = y_pred - K.mean(y_pred)

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean, axis=-1)
    d = K.sum(K.square(x_mean), axis=-1) * K.sum(K.square(y_mean), axis=-1)

    return K.mean(n / (K.sqrt(d) + 1e-12))


def pearson_loss(y_true, y_pred):
    """Loss function to maximize pearson correlation. IN PROGRESS"""

    range_var = K.reshape(K.arange(0, 7, dtype='float32'), (7, 1))
    x_mean = K.squeeze(K.dot(y_true, range_var), 2)
    y_mean = K.squeeze(K.dot(y_pred, range_var), 2)

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean, axis=-1)
    d = (K.sum(K.square(x_mean), axis=-1) *
         K.sum(K.square(y_mean), axis=-1))

    return -K.mean(n / (K.sqrt(d + 1e-12)))


def bin_percent(i):
    """Metric that keeps track of percentage of outputs in each bin."""

    def _prct(_, y_pred):
        y_pred = K.argmax(y_pred, axis=-1)

        return {str(i): K.mean(K.equal(y_pred, i))}

    return _prct


if __name__ == '__main__':
    num_timesteps = 100  # Sampling rate is 100 Hz
    nb_epoch = 10
    batch_size = 32

    def _grouper():
        iterable = utils.generate_training_set(num_timesteps=num_timesteps)
        while True:
            batched = zip(*(iterable.next() for _ in xrange(batch_size)))
            yield ([np.asarray(batched[i]) for i in range(2)],
                   [np.asarray(batched[2]),])

    dataset, calcium, spikes = utils.get_training_set(num_timesteps)

    model = build_model(num_timesteps)

    # Loss functions: Try categorical crossentropy and pearson loss.
    model.compile(optimizer='adam', loss=pearson_loss,
                  metrics=[pearson_corr] + [bin_percent(i) for i in range(7)])
    model.fit([dataset, calcium], [spikes],
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.1)

    # May be a good idea to train categorical crossentropy after training the
    # pearson correlation.
