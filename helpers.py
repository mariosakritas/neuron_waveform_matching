#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
note: STRF estimation requires the lnpy package: https://github.com/arnefmeyer/lnpy

Created on Fri March 29 15:11:11 2019

@author: arne, marios, jules
"""

from __future__ import print_function

import os
import os.path as op
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from math import log10, floor


def get_recordings(session_path, patterns=['_NoiseBurst',
                                           '_FMSweep',
                                           '_ToneSequence',
                                           '_DRC']):

    rec_dirs = []
    for d in os.listdir(session_path):
        if len([d for p in patterns if p in d]) > 0:
            rec_dirs.append(d)

    return list(sorted(set(rec_dirs)))


def get_sessions_with_drc_recordings(path):

    sessions = {}
    for root, dirs, files in os.walk(path):

        if 'advancer_data.npz' in files:

            advancer_data = np.load(op.join(root, 'advancer_data.npz'),
                                    encoding='latin1',
                                    allow_pickle=True)

            if 'channelgroup_1' in advancer_data:

                recordings = get_recordings(root,
                                            patterns=['_DRC'])
                if len(recordings) > 0:
                    parts = root.split(op.sep)
                    s = op.sep.join(parts[-3:])
                    sessions[s] = {'relative_path': s,
                                   'absolute_path': root,
                                   'recordings': recordings,
                                   'name': parts[-1],
                                   'date': parts[-2],
                                   'mouse': parts[-3]}

    return sessions


def create_filename_from_session_path(session_path):

    parts = session_path.split(op.sep)

    return parts[-3] + '_' + parts[-2] + '_' + parts[-1]

def round_sig(x, sig=3):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def load_drc_data(rec_path, max_lag=.3, max_cgf_lag=0.24):

    from lnpy.util import segment_spectrogram

    # load stimulus data from "extradata" file
    data = np.load(op.join(rec_path, 'stim_extradata.npz'))

    dt = float(data['dt'])
    n_lag = int(np.ceil(max_lag / dt))
    n_lag_cgf = int(np.ceil(max_cgf_lag / dt))
    X = data['mask']  #  / np.max(data['mask'])

    if np.max(X) > 1:
        # amplitude level with respect to 1mw  (dBm); used for transforming
        # dB-scaled data to linear scale; max(X) <= 1 assumes that the data
        # are already scaled linearly.
        ind = X > 0
        X[ind] = 10. ** ((X[ind] - np.max(X[ind])) / 20.)

    # time-lagged version (to include recent stimulus history)
    XX = segment_spectrogram(X, n_lag)

    rf_size = (n_lag, len(data['f_center']))
    trial_duration = X.shape[0] * dt

    return {'X': X,
            'XX': XX,
            'rf_size': rf_size,
            'f_center': data['f_center'],
            'dt': dt,
            'trial_duration': trial_duration,
            'n_lag': n_lag,
            'n_lag_cgf': n_lag_cgf,
            'max_lag': max_lag}


def bin_spike_train(spike_train, stim_events,
                    bin_width=.02,
                    max_lag=.3,
                    trial_duration=None):

    if trial_duration is None:
        trial_duration = round(np.mean(np.diff([ev.time for ev in stim_events])))

    n_bins = int(np.round(trial_duration / bin_width))
    n_trials = len(stim_events)
    Y = np.zeros((n_bins, n_trials))

    for i, ev in enumerate(stim_events):
        edges = ev.time + np.arange(n_bins + 1) * bin_width
        Y[:, i] = np.histogram(spike_train, bins=edges)[0]

    n_lag = int(np.ceil(max_lag / bin_width))

    return {'Y': Y,
            'y':  np.sum(Y, axis=1),
            'YY': Y[n_lag - 1:, :],
            'yy': np.sum(Y, axis=1)[n_lag - 1:]}


def bin_multiple_spike_trains(spiketrains, stim_events,
                              bin_width=.02,
                              max_lag=.3,
                              trial_duration=None):
    
    if trial_duration is None:
        trial_duration = round(np.mean(np.diff([ev.time for ev in stim_events[0]])))
        
    n_bins = int(np.round(trial_duration / bin_width))
    n_trials = [len(stim) for stim in stim_events]
    H = [np.zeros((n_bins, n)) for n in n_trials]
    tot_trials = sum(n_trials)

    for i, h in enumerate(H):
        for j, ev in enumerate(stim_events[i]):
            edges = ev.time + np.arange(n_bins + 1) * bin_width
            h[:, j] = np.histogram(spiketrains[i], bins=edges)[0]
    
    Y = np.hstack(H)
    
    n_lag = int(np.ceil(max_lag / bin_width))
    
    return {'Y': Y,
            'y':  np.sum(Y, axis=1),
            'YY': Y[n_lag - 1:, :],
            'yy': np.sum(Y, axis=1)[n_lag - 1:]}


    

class CGFModel(object):
    
    def __init__(self,
                 model = None,
                 matching_index=0
                 ):
        
        self.model = model
        self.matching_index = matching_index
          
    def as_dict(self):

        return {k: self.__dict__[k] for k in self.__dict__ if not k.startswith('_')}  
        
class SpikeUnit(object):

    def __init__(self,
                 tetrode=1,
                 cluster=0,
                 waveforms=None,
                 cluster_group='Good',
                 index=0,
                 depth=0.,
                 session=None,
                 recording_day=None,
                 date=None,
                 absolute_index=None,
                 ):

        self.tetrode = tetrode
        self.cluster = cluster
        self.waveforms = waveforms
        self.cluster_group = cluster_group
        self.index = index
        self.depth = depth
        self.session = session
        self.recording_day = recording_day
        self.date = date
        self.absolute_index = absolute_index

    def as_dict(self):

        return {k: self.__dict__[k] for k in self.__dict__ if not k.startswith('_')}


def plot_waveforms(ax, W,
                   channel_shift=None,
                   sample_rate=30000.,
                   color=3*[.2]):

    if channel_shift is None:
        channel_shift = np.max(np.abs(W))

    n_channels = W.shape[1]
    for i in range(n_channels):

        w = W[:, i]
        x = np.arange(w.shape[0]) / float(sample_rate)
        ax.plot(x, w + i*channel_shift, '-', color=color, linewidth=2)

    ax.axis('off')



def cm2inch(value):
    return value/2.54


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


class SeabornFig2Grid():
    '''Class allowing to do multiple subplots with seaborn'''
    
    
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())