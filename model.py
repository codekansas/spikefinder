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

from keras.layers import BatchNormalization
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import merge
from keras.layers import TimeDistributed

from keras.models import Model


def build_model(num_input_timesteps):
    calcium = Input(shape=(num_timesteps, 1), dtype='float32', name='calcium')
    spikes = Input(shape=(num_timesteps, 1), dtype='int32', name='spikes')

    # Embed spikes into vector space.
    flat = Flatten()(spikes)
    emb = Embedding(7, 7, init='orthogonal')(flat)

    # Normalizes the data along the time dimension.
    calcium_norm = BatchNormalization(mode=2, axis=1)(calcium)
    spikes_norm = BatchNormalization(mode=2, axis=1)(emb)

    # Like embedding the calcium channel, sort of.
    dense = Dense(128,
                  activation='relu',
                  W_regularizer='l2',
                  activity_regularizer='activity_l2')
    calcium_hid = TimeDistributed(dense)(calcium_norm)

    # Merge channels together.
    hidden = merge([calcium_hid, spikes_norm], mode='concat', concat_axis=-1)

    # Adds three convolutional layers.
    for _ in range(3):
        hidden = Convolution1D(64, 3,
                               border_mode='same',
                               activation='relu',
                               W_regularizer='l2',
                               activity_regularizer='activity_l2')(hidden)

    # Adds two recurrent layers.
    hidden = LSTM(64, return_sequences=True)(hidden)
    hidden = LSTM(64, return_sequences=False)(hidden)

    # Adds prediction layer.
    output = Dense(7, activation='softmax')(hidden)

    # Builds the model.
    model = Model(input=[calcium, spikes], output=[output])

    return model


if __name__ == '__main__':
    num_timesteps = 100
    samples_per_epoch = 1000
    nb_epoch = 10
    batch_size = 32

    def _grouper():
        iterable = utils.generate_training_set(num_timesteps)
        while True:
            batched = zip(*(iterable.next() for _ in xrange(batch_size)))
            yield ([np.asarray(batched[0]), np.asarray(batched[0])],
                   [np.asarray(batched[2]),])

    r = _grouper()

    model = build_model(num_timesteps)

    # TODO: Add precision / recall metrics?
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['acc'])

    # Defines the class weights (how much to weight each output loss).
    class_weight = {
        0: 0.1,  # No spikes, weight way less (doesn't have much information).
        1: 1.,  # Weight the other ones approximately according to their freq.
        2: 10.,
        3: 0.,  # Only worry about getting 1 and 2 correctly classified.
        4: 0.,
        5: 0.,
        6: 0.,
    }

    # TODO: The data generator should mix up the datasets better.
    # TODO: The data generator could sample more from data which ends in a
    # spike and less from data that doesn't end in a spike (this would avoid
    # having to do the class weights).
    model.fit_generator(_grouper(),
                        samples_per_epoch=samples_per_epoch,
                        nb_epoch=nb_epoch,
                        class_weight=class_weight)
