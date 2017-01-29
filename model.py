"""Stacked CNN + RNN that predicts spikes given calcium recordings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import utils

from keras.callbacks import ModelCheckpoint

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
from keras.models import load_model


def conv_bn(x, nb_filter, filter_length):
    """Applies convolution and batch normalization."""

    x = Convolution1D(nb_filter, filter_length,
                      activation='relu',
                      border_mode='same')(x)
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

    # Adds some more features from the calcium data.
    calcium_bn = BatchNormalization(axis=1)(calcium)
    calcium_sq = utils.QuadFeature()(calcium_bn)

    delta_1 = utils.DeltaFeature()(calcium_bn)
    delta_1_sq = utils.QuadFeature()(delta_1)

    delta_2 = utils.DeltaFeature()(delta_1)
    delta_2_sq = utils.QuadFeature()(delta_2)

    calcium_input = merge([calcium_bn, calcium_sq,
                           delta_1, delta_1_sq,
                           delta_2, delta_2_sq],
                          mode='concat', concat_axis=2)
    calcium_norm = BatchNormalization(axis=1)(calcium_input)

    # Embed dataset into vector space.
    flat = Flatten()(RepeatVector(num_timesteps)(dataset))
    data_emb = Embedding(10, 1, init='orthogonal')(flat)

    # Merge channels together.
    hidden = merge([calcium_input, data_emb],
                   mode='concat', concat_axis=-1)

    # Adds convolutional layers.
    for _ in range(3):
        hidden = inception_cell(hidden)

    # Adds recurrent layers.
    # hidden = Bidirectional(LSTM(64, return_sequences=True))(hidden)
    # hidden = Bidirectional(LSTM(64, return_sequences=True))(hidden)

    # Adds output layer.
    output = TimeDistributed(Dense(1, activation='softplus'))(hidden)

    # Builds the model.
    model = Model(input=[dataset, calcium], output=[output])

    return model


if __name__ == '__main__':
    # Sampling rate is 100 Hz.
    num_timesteps = 100

    nb_epoch = 10
    batch_size = 32
    model_save_loc = '/tmp/best.keras_model'
    output_save_loc = '/tmp/'
    train_on_subset = True  # Set this to train on a small subset of the data.

    def _grouper():
        iterable = utils.generate_training_set(num_timesteps=num_timesteps)
        while True:
            batched = zip(*(iterable.next() for _ in xrange(batch_size)))
            yield ([np.asarray(batched[i]) for i in range(2)],
                   [np.asarray(batched[2]),])

    model = build_model(num_timesteps)

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

    dataset, calcium, spikes = utils.get_training_set(num_timesteps)

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

    metrics = [utils.pearson_corr]
    metrics += [utils.bin_percent(i) for i in range(7)]

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
