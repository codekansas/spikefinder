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


def spike_hist():
    """Generates a histogram of spike counts over all the data."""

    import matplotlib.pyplot as plt

    plt.figure()
    for i, (_, spikes) in enumerate(utils.get_data_set('train')):
        x = np.reshape(spikes, (-1,))
        x = x[np.isnan(x) == False]
        print(x.shape)

        plt.subplot(5, 2, i + 1)
        plt.hist(np.cast[np.int32](x), range(6), log=True)

    plt.show()


if __name__ == '__main__':
    spike_hist()

#     m = 0
#     for i, (calcium, spikes, last_spike) in enumerate(utils.generate_training_set()):
#         # print(i, calcium.shape, spikes.shape, last_spike.shape)
#         m = max(m, int(last_spike))
#         print(i, m)
