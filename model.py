"""Stacked CNN + RNN that predicts spikes given calcium recordings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import utils

import keras.backend as K
from keras.callbacks import ModelCheckpoint

from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers import merge
from keras.layers import RepeatVector
from keras.layers import Reshape
from keras.layers import PReLU
from keras.layers import ParametricSoftplus
from keras.layers import TimeDistributed

from keras.models import Model
from keras.models import load_model


def conv_bn(x, nb_filter, filter_length):
    """Applies convolution and batch normalization."""

    x = Convolution1D(nb_filter, filter_length,
                      activation=None,
                      border_mode='same')(x)
    x = PReLU()(x)
    x = BatchNormalization(axis=2)(x)
    return x


def inception_cell(x_input):
    """Applies a single inception cell."""

    x = x_input

    a = conv_bn(x, 64, 1)

    b = conv_bn(x, 48, 1)
    b = conv_bn(b, 64, 10)

    c = conv_bn(x, 64, 1)
    c = conv_bn(c, 96, 7)
    c = conv_bn(c, 96, 7)

    d = AveragePooling1D(7, stride=1, border_mode='same')(x)
    d = conv_bn(d, 32, 1)

    x = merge([a, b, c, d],
              mode='concat', concat_axis=-1)

    return x


def build_model(num_timesteps):
    dataset = Input(shape=(1,), dtype='int32', name='dataset')
    calcium = Input(shape=(num_timesteps, 1), dtype='float32', name='calcium')

    # Adds some more features from the calcium data.
    delta_1 = utils.DeltaFeature()(calcium)
    delta_2 = utils.DeltaFeature()(delta_1)

    calcium_input = merge([calcium, delta_1, delta_2],
                          mode='concat', concat_axis=2)

    # Embed dataset into vector space.
    flat = Flatten()(RepeatVector(num_timesteps)(dataset))
    data_emb = Embedding(10, 1, init='orthogonal')(flat)

    # Merge channels together.
    x = merge([calcium_input, data_emb],
               mode='concat', concat_axis=-1)

    x = Convolution1D(128, 1,
            init='glorot_normal',
            border_mode='same',
            activation='relu')(x)

    # Adds recurrent layers.
    # x = Bidirectional(LSTM(64,
    #         forget_bias_init='zero',
    #         return_sequences=True))(x)
    # x = Bidirectional(LSTM(64,
    #         forget_bias_init='zero',
    #         activation='relu',
    #         return_sequences=True))(x)

    # Adds convolutional layers.
    for i in range(5):
        x = inception_cell(x)

    # Adds residual layers.
    x = Convolution1D(64, 5,
            activation='relu',
            border_mode='same')(x)
    for i in range(5):
        x_c = Convolution1D(64, 5,
                activation='relu',
                border_mode='same')(x)
        # x_c = Dropout(0.3)(x_c)
        x = merge([x, x_c], mode='sum')

    x = Convolution1D(128, 1, activation='tanh')(x)
    x = Dropout(0.5)(x)

    x = Convolution1D(1, 3,
        activation='hard_sigmoid',
        border_mode='same',
        init='glorot_normal')(x)
    # x = ParametricSoftplus()(x)

    def _normalize(i):
        min_v = K.min(i)
        max_v = K.max(i)
        return (i - min_v) * 6 / (max_v - min_v)

    x = Lambda(_normalize)(x)

    # Builds the model.
    model = Model(input=[dataset, calcium], output=[x])

    return model


if __name__ == '__main__':
    # Sampling rate is 100 Hz.
    num_timesteps = 1000
    rebuild_data = False
    rebuild_model = True

    nb_epoch = 500
    batch_size = 16
    model_save_loc = ('/home/judgingmoloch/workspace/'
                      'spikefinder/data/best.keras_model')
    output_save_loc = ('/home/judgingmoloch/workspace/'
                       'spikefinder/data/outputs')
    train_on_subset = False  # Set this to train on a small subset of the data.

    def _grouper():
        iterable = utils.generate_training_set(num_timesteps=num_timesteps)
        while True:
            batched = zip(*(iterable.next() for _ in xrange(batch_size)))
            yield ([np.asarray(batched[i]) for i in range(2)],
                   [np.asarray(batched[2]),])

    model = build_model(num_timesteps)

    if not rebuild_model and os.path.exists(model_save_loc):
        print('Loading weights from "%s".' % model_save_loc)
        model.save_weights(model_save_loc)

    def _save_predictions(model, dataset):
        """Saves the predictions of the model."""

        for d_idx, output_shape, it in utils.get_eval(dataset):
            file_name = '%d.%s.spikes.csv' % (d_idx + 1, dataset)
            file_path = os.path.join(output_save_loc, file_name)
            tmp_path = os.path.join(output_save_loc, 'tmp_' + file_name)

            # Initializes a NaN array to store the outputs.
            arr = np.empty(output_shape)
            arr[:] = np.NAN

            for c_idx, data_len, data in it:
                print('%d/%d' % (c_idx + 1, output_shape[1]))
                d_v = np.cast['int32'](np.ones((data.shape[0], 1)) * d_idx)
                model_preds = model.predict([d_v, data],
                                            verbose=1,
                                            batch_size=100)
                model_preds = np.reshape(model_preds, (-1,))
                model_preds = utils.output_to_ints(model_preds[:data_len])
                arr[:model_preds.shape[0], c_idx] = model_preds

            np.savetxt(tmp_path,
                       arr,
                       fmt='%.0f',
                       delimiter=',',
                       header=','.join(str(i) for i in range(output_shape[1])),
                       comments='')

            # Replaces NaNs with empty.
            with open(tmp_path, 'rb') as fin:
                with open(file_path, 'wb') as fout:
                    for line in fin:
                        fout.write(line.replace('nan', ''))

            print('Saved "%s".' % file_path)

    dataset, calcium, spikes = utils.get_training_set(num_timesteps,
            rebuild=rebuild_data)

    if train_on_subset:
        idx = np.random.choice(np.arange(dataset.shape[0]), size=5000)
        dataset = dataset[idx]
        calcium = calcium[idx]
        spikes = spikes[idx]

    # Save the model with the best validation Pearson correlation.
    save_callback = ModelCheckpoint(model_save_loc,
                                    monitor='val_pearson_corr',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max')

    metrics = [utils.pearson_corr, utils.stats]
    # metrics += [utils.bin_percent(i) for i in range(7)]

    # Loss functions: Try categorical crossentropy and pearson loss.
    model.compile(optimizer='adam',
                  loss=utils.pearson_loss,
                  metrics=metrics)
    model.fit([dataset, calcium], [spikes],
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.1,
              callbacks=[save_callback])

    # Saves the best model predictions on the training set.
    model.load_weights(model_save_loc)
    _save_predictions(model, 'train')
    _save_predictions(model, 'test')
