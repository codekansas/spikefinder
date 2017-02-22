#!/usr/bin/env python
"""Stacked CNN + RNN that predicts spikes given calcium recordings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import utils

import keras.backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from keras.layers import Activation
from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import Cropping1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import merge
from keras.layers import RepeatVector
from keras.layers import Reshape
from keras.layers import PReLU
from keras.layers import ParametricSoftplus
from keras.layers import TimeDistributed

from keras.models import Model
from keras.models import load_model


def build_model(num_timesteps,
        buffer_length,
        use_dataset,
        use_calcium_stats):
    calcium = Input(shape=(num_timesteps, 1), dtype='float32', name='calcium')
    inputs = [calcium]

    if use_calcium_stats:
        calcium_stats = Input(shape=(num_timesteps, 6), dtype='float32')
        inputs.append(calcium_stats)
        x = merge([calcium, calcium_stats], mode='concat')
    else:
        x = calcium

    # Adds some more features.
    delta_1 = utils.DeltaFeature()(x)
    delta_2 = utils.DeltaFeature()(delta_1)
    quad_1 = utils.QuadFeature()(x)
    quad_2 = utils.QuadFeature()(delta_1)
    quad_3 = utils.QuadFeature()(delta_2)

    # Merge channels together.
    x = merge([x, delta_1, delta_2, quad_1, quad_2, quad_3],
               mode='concat', concat_axis=-1)
    # x = BatchNormalization(axis=1)(x)  # normalize across time.

    # Extracts first-level (dataset-independent) features.
    x = Convolution1D(32, 2,
            init='glorot_normal',
            border_mode='same',
            activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = BatchNormalization(axis=1)(x)  # normalize across time.

    if use_dataset:
        dataset = Input(shape=(1,), dtype='int32', name='dataset')
        inputs.append(dataset)
        d_emb = Flatten()(Embedding(10, 32)(dataset))
        d_emb = Activation('tanh')(d_emb)
        x = Lambda(lambda x: x * K.expand_dims(d_emb, 1))(x)

    # x = LSTM(512, return_sequences=True, forget_bias_init='one')(x)

    # Given weighed first-level features, look for second-level features.
    x = Convolution1D(64, 16,
        activation='relu',
        border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)  # normalize across time.

    x = Convolution1D(128, 8,
        activation='relu',
        border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)  # normalize across time.

    x = Convolution1D(256, 4,
        activation='relu',
        border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)  # normalize across time.

    x = Convolution1D(512, 2,
        activation='relu',
        border_mode='same')(x)
    x = BatchNormalization(axis=1)(x)  # normalize across time.

    x = Convolution1D(1024, 1,
        activation='relu',
        border_mode='same')(x)
    x = Dropout(0.5)(x)

    # x = Convolution1D(512, 1,
    #     activation='tanh',
    #     border_mode='same')(x)
    # x = Dropout(0.5)(x)

    x = Convolution1D(1, 1,
        activation=None,
        border_mode='same',
        W_regularizer='l2',
        init='glorot_normal')(x)

    x = Cropping1D((buffer_length, buffer_length))(x)

    # Builds the model.
    model = Model(input=inputs, output=[x])

    return model


def evaluate(model, args, mode='train'):
    """Evaluates and saves to CSV."""

    raise NotImplementedError('TODO')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Spikefinder util function',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--num-timesteps',
            default=1000,
            type=int,
            help='number of timesteps')
    parser.add_argument('-d', '--rebuild-data',
            default=False,
            action='store_true',
            help='if set, rebuild the dataset')
    parser.add_argument('-m', '--rebuild-model',
            default=False,
            action='store_true',
            help='if set, ignore saved weights')
    parser.add_argument('-n', '--num-epochs',
            default=50,
            type=int,
            help='number of epochs to train')
    parser.add_argument('-p', '--plot',
            default=False,
            action='store_true',
            help='if set, plot a sample prediction')
    parser.add_argument('-b', '--batch-size',
            default=8,
            type=int,
            help='size of each minibatch')
    parser.add_argument('--model-location',
            default='/tmp/best.keras_model',
            type=str,
            help='where to save the model')
    parser.add_argument('--output-location',
            default='/tmp/data_outputs',
            type=str,
            help='where to save the outputs')
    parser.add_argument('--buffer-length',
            default=100,
            type=int,
            help='amount to buffer at beginning and end')
    parser.add_argument('--ignore-dataset',
            default=False,
            action='store_true',
            help='if set, ignore the dataset as a feature')
    parser.add_argument('--ignore-calcium-stats',
            default=False,
            action='store_true',
            help='if set, ignore the batch calcium statistics')
    parser.add_argument('-l', '--loss',
            default='mse',
            type=str,
            choices=['crossentropy', 'pearson', 'mse'],
            help='type of loss function to use')
    parser.add_argument('-v', '--num_val',
            default=300,
            type=int,
            help='number of validation samples')
    parser.add_argument('-e', '--evaluate',
            default=False,
            action='store_true',
            help='if set, evaluate on the testing data')

    args = parser.parse_args()

    # Builds the model.
    model = build_model(args.num_timesteps,
            args.buffer_length,
            use_dataset=not args.ignore_dataset,
            use_calcium_stats=not args.ignore_calcium_stats)

    # Handles model loading / rebuilding.
    if args.rebuild_model:
        if os.path.exists(args.model_location):
            os.remove(args.model_location)
    else:
        if os.path.exists(args.model_location):
            model.load_weights(args.model_location)
            print('Loaded weights from "%s".' % args.model_location)
        else:
            print('No weights found at "%s".' % args.model_location)

    # Gets the training data.
    dataset, calcium, calcium_stats, spikes = utils.get_training_set(
            buffer_length=args.buffer_length,
            num_timesteps=args.num_timesteps,
            rebuild=args.rebuild_data)

    # Cuts off the beginning and end bits.
    spikes = spikes[:, args.buffer_length:-args.buffer_length]

    # Builds the inputs according to the user's specifications.
    inputs = [calcium]
    if not args.ignore_calcium_stats:
        inputs.append(calcium_stats)
    if not args.ignore_dataset:
        inputs.append(dataset)

    # Splits into training and validation sets.
    def _split(x):
        x = zip(*[(i[:-args.num_val], i[-args.num_val:]) for i in x])
        x = [list(i) for i in x]
        return x

    inputs, val_inputs = _split(inputs)
    spikes, val_spikes = _split([spikes])

    # Save the model with the best validation Pearson correlation.
    save_callback = ModelCheckpoint(args.model_location,
                                    monitor='val_pearson_corr',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max')

    # Keep track of pearson correlation.
    metrics = [utils.pearson_corr, utils.stats]

    # Loss functions: Try crossentropy and pearson loss.
    if args.loss == 'crossentropy':
        loss = 'binary_crossentropy'
    elif args.loss == 'pearson':
        loss = utils.pearson_loss
    elif args.loss == 'mse':
        loss = 'mse'
    else:
        raise ValueError('Invalid loss: "%s".' % args.loss)

    # Compiles and trains the model.
    model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss=loss,
                  metrics=metrics)
    model.fit(inputs, spikes,
              batch_size=args.batch_size,
              nb_epoch=args.num_epochs,
              validation_data=[val_inputs, val_spikes],
              callbacks=[save_callback])

    # Loads the best model predictions on the training set.
    if os.path.exists(args.model_location):
        model.load_weights(args.model_location)

    # Plots samples of the model's performace.
    if args.plot:
        import matplotlib.pyplot as plt

        x = np.arange(0, args.num_timesteps) / 100
        x_buf = x[args.buffer_length:-args.buffer_length]

        plt.figure()

        # For scaling inputs to [0, 1].
        _scale = lambda x: (x - np.min(x)) * (np.max(x) - np.min(x) + 1e-12)

        for i in range(3):
            idx = np.random.randint(args.num_val)
            pred_on = [val_input[idx:idx+1] for val_input in val_inputs]
            preds = model.predict(pred_on)

            # Plots the spikes and spike predictions.
            ax = plt.subplot(2, 3, i + 1)
            plt.plot(x_buf, preds[0], label='Predictions')
            # plt.plot(x_buf, np.floor(preds[0]),
            #         label='Predictions (Floored)')
            plt.plot(x_buf, val_spikes[0][idx],
                    label='Actual Spikes')
            plt.xlabel('time (s)')
            plt.legend()

            # Plots the calcium trace.
            plt.subplot(2, 3, i + 4, sharex=ax)
            plt.plot(x, val_inputs[0][idx],
                    label='Calcium Trace')
            plt.xlabel('time (s)')
            plt.legend()

        plt.show()

    # Runs the evaluation script.
    if args.evaluate:
        evaluate(model, args)
