#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

example of bar plot with signal data points

Created on Wed Aug 29 12:36:22 2018

@author: arne
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    conditions = ['RandReg', 'RegRand', 'Step']
    n_trials = 50

    fig, ax = plt.subplots()

    # manually add y=0 line
    ax.axhline(0, lw=1, color=3*[0])

    n_conditions = len(conditions)
    for i in range(n_conditions):

        # simulate some data points with mean i and unit standard deviation
        y = i + np.random.randn(n_trials)

        # single data points
        ax.plot(i*np.ones_like(y), y, 'o',
                color=3*[.25],
                ms=4,
                mec='none')

        # bar showing mean + std
        ax.bar(i, np.mean(y),
               width=.5,
               align='center',
               color='#1f77b4',  # 3*[.5],
               alpha=.5,
               edgecolor=3*[.1],
               linewidth=1,
               yerr=np.std(y)/np.sqrt(n_trials),
               capsize=8)

    ax.set_xlabel('')
    ax.set_xlim(-.5, n_conditions-.5)
    ax.set_xticks(range(n_conditions))
    ax.set_xticklabels(conditions)

    ax.set_ylabel('Relative change\nfrom baseline (%)', labelpad=-8)
    ax.set_ylim(-10, 10)
    ax.set_yticks([-10, 0, 10])

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # hide bottom ticks
    ax.tick_params(which='major', axis='x', length=0)

    fig.set_size_inches(4, 2.5)
    fig.tight_layout(pad=.5)

    plt.show()
