#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:22:08 2018

@author: arne
"""

import numpy as np
from scipy import signal
import os.path as op
import sys
import matplotlib.pyplot as plt


def filter_data(x, fs, f_cutoff=5., order=2):

    Wn = f_cutoff / fs * 2
    b, a = signal.butter(order, Wn,
                         btype='lowpass',
                         analog=False,
                         output='ba')

    data = signal.filtfilt(b, a, x, axis=0)

    return data


def load_cylinder_running_data(rec_path,
                               binwidth=0.2,  # bin width for TTL pulses
                               t_range=None,  # time range for running
                               f_smoothing=1.  # cutoff frequency for smoothing
                               ):

    # diameter of cylinder is 20 cm; the rotatry encoder has 1024 steps per
    # full rotation
    scale_factor = np.pi * 20. / 1024

    # extract locomotion data from rotary encoder TTL pulses
    # channel 5: backward
    # channel 6: forward
    data = np.load(op.join(rec_path, 'ttl_events.npz'), encoding="latin1")

    v1 = data['channels'] == 5  # backward
    v2 = data['channels'] == 6  # forward
    ts1 = data['timestamps'][v1]
    ts2 = data['timestamps'][v2]

    # check time range
    if t_range is None:
        t_min, t_max = data['timestamps'].min(), data['timestamps'].max()
    else:
        t_min, t_max = t_range

    # convert to "rate" using histograms
    T = t_max - t_min
    n_bins = int(round(T / float(binwidth) + .5))
    bins = t_min + np.arange(n_bins) * binwidth
    cnt1 = np.histogram(ts1, bins=bins)[0] * scale_factor
    cnt2 = np.histogram(ts2, bins=bins)[0] * scale_factor

    # low-pass filtered histograms
    backward = filter_data(cnt1, 1./binwidth, f_cutoff=f_smoothing)
    forward = filter_data(cnt2, 1./binwidth, f_cutoff=f_smoothing)

    # use center of time bins as reference time
    t = .5*(bins[:-1] + bins[1:])

    return t, backward, forward


if __name__ == '__main__':

    t, backward, forward = load_cylinder_running_data(sys.argv[1])

    fig, ax = plt.subplots()

    ax.plot(t, backward, 'k-', label='backward')
    ax.plot(t, forward, 'r-', label='forward')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Running speed (cm/s)')

    ax.legend(loc='best')

    plt.show()
