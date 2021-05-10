#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
simple example script illustrating generation of Chait lab stimuli

Created on Wed Mar 21 15:34:41 2018

@author: arne
"""

import matplotlib.pyplot as plt
import os.path as op
import sys
sys.path.append(op.join(op.split(__file__)[0], '..'))

from randreg.chait import ChaitStimulusGenerator   


def run_example():

    fs = 48000.
    stimgen = ChaitStimulusGenerator(samplerate=fs,
                                     n_trials=5,
                                     create_events=True,
                                     create_extra_data=True)
    X, events, extra_data = stimgen.generate()

    plt.specgram(X.ravel(), NFFT=256, Fs=fs, noverlap=128)
    plt.show()


if __name__ == '__main__':
    run_example()
