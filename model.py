"""Model definitions.

The model takes as input the calcium channel recordings and number of spikes
over some amount of time and tries to predict the number of spikes on the last
interval of the recording.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils


if __name__ == '__main__':
    for i, (calcium, spikes, last_spike) in enumerate(utils.generate_training_set()):
        print(i, calcium.shape, spikes.shape, last_spike.shape)
