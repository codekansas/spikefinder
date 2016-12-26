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

    l = list()
    for _, spikes in utils.get_data_set('train'):
        x = np.reshape(spikes, (-1,))
        x = x[np.isnan(x) == False]
        print(x.shape)
        l.append(x)
    c = np.concatenate(l, 0)

    plt.hist(np.cast[np.int32](c), 6, log=True)
    plt.show()


if __name__ == '__main__':
    m = 0
    for i, (calcium, spikes, last_spike) in enumerate(utils.generate_training_set()):
        # print(i, calcium.shape, spikes.shape, last_spike.shape)
        m = max(m, int(last_spike))
        print(i, m)
