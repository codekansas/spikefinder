#!/usr/bin/env python
# Quick script for evaluting everything.

from __future__ import print_function

from spikefinder import load, score
import numpy as np

for i in range(1, 11):
    name_1 = '/tmp/%d.train.spikes.csv' % i
    name_2 = 'data/spikefinder.train/%d.train.spikes.csv' % i
    a = load(name_1)
    b = load(name_2)
    s = score(a, b)
    print('train %d: [' % i,
          'mean = %f,' % np.nanmean(s),
          'median = %f' % np.nanmedian(s),
          ']')

