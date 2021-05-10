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
from scipy.interpolate import interp1d
import glob
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from math import log10, floor


def get_custom_colormap(colormap):
    
    nbins = 64
    if colormap=='prf':
        colors = [(0.1, 0.25, 0.4), (1, 1, 1), (0.8, 0.6, 0.4)]

    elif colormap=='cgf':
        colors = [(0, 0, 0.5), (1, 1, 1), (0.5, 0, 0)]
        
    cmap_name = colormap
    cm = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=nbins)
    
    return cm

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

def bin_two_spike_trains(spike_train_1, stim_events_1, spike_train_2, stim_events_2,
                    bin_width=.02,
                    max_lag=.24,
                    trial_duration=None):

    if trial_duration is None:
        trial_duration = round(np.mean(np.diff([ev.time for ev in stim_events_1])))

    n_bins = int(np.round(trial_duration / bin_width))
    n_trials_1 = len(stim_events_1)
    H =  np.zeros((n_bins, n_trials_1))
    n_trials_2 = len(stim_events_2)
    G =  np.zeros((n_bins, n_trials_2))
    n_trials = n_trials_1+ n_trials_2
    Y = np.zeros((n_bins, n_trials))

    for i, ev in enumerate(stim_events_1):
        edges = ev.time + np.arange(n_bins + 1) * bin_width
        H[:, i] = np.histogram(spike_train_1, bins=edges)[0]
    for i, ev in enumerate(stim_events_2):
        edges = ev.time + np.arange(n_bins + 1) * bin_width
        G[:, i] = np.histogram(spike_train_2, bins=edges)[0]

    Y = np.hstack((H,G))

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


# ----------------------------------------------------------------------------
# Behavioral analysis
# ----------------------------------------------------------------------------


def filter_data(x, fs, f_cutoff=5., order=2):

    Wn = f_cutoff / fs * 2
    b, a = signal.butter(order, Wn,
                         btype='lowpass',
                         analog=False,
                         output='ba')

    data = signal.filtfilt(b, a, x, axis=0)

    return data


def detect_threshold_crossings(ts, env,
                               threshold=0.5,
                               min_still=0.5,
                               max_still=np.Inf,
                               min_duration=.5,
                               threshold_type='lower'):
    # get "onset" (and "offset") time signal (using thresholding)

    # get binary mask indicating movement/no movement
    if threshold_type == 'lower':
        mask = np.asarray(env > threshold, dtype=np.int)

    elif threshold_type == 'upper':
        mask = np.asarray(env <= threshold, dtype=np.int)

    # extract onset and offset times
    onset_times = ts[np.where(np.diff(mask) > 0)[0]]
    offset_times = ts[np.where(np.diff(mask) < 0)[0]]

    if onset_times.shape[0] > 0 and offset_times.shape[0] > 0:

        offset_times = offset_times[offset_times > onset_times[0]]

        if len(onset_times) > len(offset_times):
            onset_times = onset_times[:offset_times.shape[0]]

        v = np.ones((onset_times.shape[0],), dtype=np.bool)
        if min_still > 0 and max_still < np.Inf:
            # check for offset-onset intervals in given range;
            # for simplicity always include first onset
            dT = offset_times[:-1] - onset_times[1:]
            v[1:] = np.logical_and(dT >= min_still, dT <= max_still)

        # check for minimum onset duration
        dT = offset_times - onset_times
        v = np.logical_and(v, dT >= min_duration)

        print("Ignored", np.sum(v == 0), "out of", v.shape[0], "onsets")

        # create matrix with onset and offset times as columns
        T = np.column_stack((onset_times, offset_times))
        T = T[v, :]

        # # update mask
        # mask[:] = 0
        # for tt in T:
        #     v = np.logical_and(ts >= tt[0], ts <= tt[1])
        #     mask[v] = 1

    else:
        T = np.zeros((0, 2))

    return T


def get_pupil_files(rec_path,
                    camera='*'):

    pupil_files = glob.glob(op.join(rec_path,
                                    'rpi_camera_{}_pupil_data.npy'.format(camera)))
    return pupil_files


def smooth_traces_for_plotting(y, n=3,
                               interpolate=True,
                               verbose=False):

    v1 = ~np.isnan(y)

    from scipy.signal import gaussian
    win = gaussian(n, .15*n)
    y = np.convolve(y, win/win.sum(), 'same')

    if interpolate:

        v2 = ~np.isnan(y)
        vv = np.logical_and(v1, ~v2)
        nv = np.sum(vv)
        if nv:

            if verbose:
                print("smooth_traces_for_plotting, interpolating samples:", nv)

            x = np.arange(y.shape[0])
            f = interp1d(x[~vv], y[~vv],
                         bounds_error=False,
                         fill_value=np.NaN)
            y[vv] = f(x[vv])

    return y


def check_recording_contains_running_data(rec_path):

    # load TTL pulses generated by the rotary encoder attached
    # to the styrofoam cylinder
    ttl = np.load(op.join(rec_path, 'ttl_events.npz'),
                  encoding='latin1',
                  allow_pickle=True)

    status = False
    if 5 in ttl['channels'] or 6 in ttl['channels']:
        status = True

    return status


def load_cylinder_running_data(rec_path,
                               binwidth=0.2,  # bin width for TTL pulses
                               t_range=None,  # time range for running
                               smooth=True,
                               f_smoothing=1.  # cutoff frequency for smoothing
                               ):

    # diameter of cylinder is 20 cm; the rotatory encoder has 1024 steps per
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

    if f_smoothing > 0:
        # low-pass filtered histograms
        backward = filter_data(cnt1, 1./binwidth, f_cutoff=f_smoothing)
        forward = filter_data(cnt2, 1./binwidth, f_cutoff=f_smoothing)
    else:
        backward = cnt1
        forward = cnt2

    # use center of time bins as reference time
    t = .5*(bins[:-1] + bins[1:])

    # compute speed (ignoring direction)
    speed = np.sqrt(backward ** 2 + forward ** 2)

    return t, backward, forward, speed


def extract_running_bouts(ts, speed, threshold=2, min_still=0.5, max_still=np.Inf,
                          min_duration=.5, verbose=True):

    # get binary mask indicating running/no running
    mask = np.asarray(speed > threshold, dtype=np.int)

    # running onset and offset times
    onset_times = ts[np.where(np.diff(mask) > 0)[0]+1]
    offset_times = ts[np.where(np.diff(mask) < 0)[0]]

    if onset_times.shape[0] > 0 and offset_times.shape[0] > 0:

        # only include offset times following an onset
        offset_times = offset_times[offset_times > onset_times[0]]

        if len(onset_times) > len(offset_times):
            onset_times = onset_times[:offset_times.shape[0]]

        # check for offset-onset invervals in given range
        dt = onset_times[1:] - offset_times[:-1]

        v = np.ones((onset_times.shape[0],), dtype=np.bool)
        v[1:] = np.logical_and(dt >= min_still, dt <= max_still)

        # check for minimum running bout duration
        dt = offset_times - onset_times
        v = np.logical_and(v, dt >= min_duration)

        if verbose:
            print("Ignored", np.sum(v == 0), "out of", v.shape[0], "running bouts")

        # create matrix with onset and offset times as columns
        on_off_times = np.column_stack((onset_times, offset_times))
        on_off_times = on_off_times[v, :]

    else:
        on_off_times = np.zeros((0, 2))

    return on_off_times


def show(w_cgf, J, M, N, dt=0.02, cmap='seismic', show_now=True,
         v_max_cgf=None, timeticks_cgf=None, **kwargs):


    if v_max_cgf is None:
        v_max_cgf = np.max(np.abs(w_cgf[::-1, :]))

    fig = plot_context_model(w_cgf,
                             J, M, N,
                             v_max_cgf,
                             dt=dt,
                             cmap=cmap,
                             timeticks_cgf=timeticks_cgf,
                             **kwargs)

    if show_now:
        plt.show()

    return fig


def plot_context_model(w_cgf, J, M, N,
                       v_max_cgf,
                       dt=0.02,
                       cmap='RdBu_r',
                       timeticks_cgf=None,
                       colorbar=True,
                       colorbar_num_ticks=3,
                       frequencies=None,
                       freqticks=[4000, 8000, 16000, 32000],
                       logfscale=True,
                       windowtitle=None,
                       axes=None,
                       no_labels = None):
    if np.any(w_cgf) > 0 :
        w_cgf = w_cgf[::-1, :]
    else:
        print('WARNING *********THIS MEAN CGF IS EMPTY***********')

    if frequencies is None:
        frequencies = 2. ** np.arange(w_cgf.shape[1])
    w_cgf[-1, 12] = 0.0
    # frequency axis
    fc = frequencies
    df_log = np.diff(np.log2(fc))
    # n_octaves = round(N * np.mean(df_log) * 10) / 10
    n_octaves = 1.0
    #    w_cgf[-1, N] = 0
    n_plots = 1 # 2 + plot_STRF

    # Plot RFs
    new_figure = False
    if axes is None:
        new_figure = True
        fig, axes = plt.subplots(nrows=1, ncols=n_plots)
    else:
        axes = axes

    plt_args = dict(interpolation='nearest',
                    aspect='auto',
                    origin='lower')
    plot_index = 0
    images = []

    if fc is not None:

        fc = np.asarray(fc) / 1000.
        f_unit = 'kHz'

        if freqticks is not None:
            freqticks = np.asarray(freqticks) / 1000.

        # set ticks
        if logfscale:
            f_extent = [.5, fc.shape[0] + .5]
        else:
            f_extent = [fc.min(), fc.max()]

    else:
        f_extent = [.5, w_cgf.shape[1] + .5]
        f_unit = 'channels'

    # ax = axes[plot_index]
    # plot_index += 1
    # ax.set_title('PRF')
    # v_max = v_max_prf
    # v_eps = max(1e-3 * v_max, 1e-12)
    # extent = (-J * dt * 1000, 0, f_extent[0], f_extent[1])
    # im = ax.imshow(w_prf.T,
    #                vmin=-v_max - v_eps,
    #                vmax=v_max + v_eps,
    #                cmap='PuOr_r',
    #                extent=extent,
    #                **plt_args)
    # ax.set_xlabel('Time (ms)')
    # ax.set_ylabel('Frequency (%s)' % f_unit)
    # images.append(im)

    # # Set ticks for STRF and PRF
    # for ax in axes[:1 + plot_STRF]:
    #     if timeticks_prf is not None:
    #         ax.set_xticks(timeticks_prf)
    #     else:
    #         ax.xaxis.set_major_locator(MaxNLocator(4))

    ax = axes
    #ax.set_title('mean CGF')
    v_max = 0.2
    v_eps = max(1e-3 * v_max, 1e-12)
    extent = (-M * dt * 1000, 0, -1 , 1)
    im = ax.imshow(w_cgf.T,
                   vmin=-v_max - v_eps,
                   vmax=v_max + v_eps,
                   cmap=get_custom_colormap('cgf'),
                   extent=extent,
                   **plt_args)

    if no_labels:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        plt.axis('off')
        plt.tick_params(
        axis='both',  # changes apply to both axes
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    else:
        ax.set_xlabel('Time shift (ms)')
        ax.set_ylabel(r'$\Phi$ (oct)')
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([-n_octaves, 0, n_octaves])

        if timeticks_cgf is not None:
            ax.set_xticks(timeticks_cgf)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(4))

    if windowtitle is not None:
        fig.canvas.set_window_title(windowtitle)

    if colorbar:
        ax = axes #for im, ax in zip(images, axes.tolist()):
        cbar = plt.colorbar(mappable=im, ax=ax)
        cbar.locator = MaxNLocator(colorbar_num_ticks)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=5,
                            pad=2)
    if new_figure:
        fig.set_size_inches(4, 2.5)
        fig.tight_layout()

    return im


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
    '''Class allowing to to multiple subplots with seaborn'''
    
    
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