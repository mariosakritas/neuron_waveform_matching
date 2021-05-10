#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3
"""
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op
import click
import traceback
from scipy import signal
from scipy import interpolate


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

    # low-pass filtered histograms
    backward = filter_data(cnt1, 1./binwidth, f_cutoff=f_smoothing)
    forward = filter_data(cnt2, 1./binwidth, f_cutoff=f_smoothing)

    # use center of time bins as reference time
    t = .5*(bins[:-1] + bins[1:])

    # compute speed (ignoring direction)
    speed = np.sqrt(backward**2 + forward**2)

    return t, backward, forward, speed


def load_stim_events(rec_path):

    event_data = np.load(op.join(rec_path, 'stim_events.npz'),
                         encoding='latin1')
    stim_events = event_data['StimEvents'].tolist()
    conditions = list(set([ev['Stim'] for ev in stim_events]))

    return stim_events, conditions


def load_pupil_data(rec_path, smooth_len=.3):

    # load pupil data
    pupil_file = op.join(rec_path, 'rpi_camera_2_pupil_data.npz')
    pupil_data = np.load(pupil_file, encoding='latin1')
    pupil_size = np.max(pupil_data['size'], axis=1)

    # camera timestamps; should be one per frame but sometimes the last few
    # triggers were not received by open-ephys
    cam = np.load(op.join(rec_path, 'rpi_camera_2.npz'),
                  encoding='latin1')
    ts = cam['timestamps']
    N = min(pupil_size.shape[0], ts.shape[0])
    ts = ts[:N]
    pupil_size = pupil_size[:N]

    # smooth pupil trace (needs more work)
    dt = np.median(np.diff(ts))
    n_win = max(int(round(smooth_len / dt)), 1)
    if n_win % 2 == 0:
        # make sure window is symmetric
        n_win += 1
    win = signal.gaussian(n_win, 1.5)

    pupil_size = np.convolve(pupil_size, win/win.sum(), 'same')

    return ts, pupil_size


def load_data_recording(rec_path,
                        alignment='onset',
                        t_pre=2.,
                        t_post=2.,
                        dt=.05,
                        t_norm=2.,
                        normalize=True,
                        running_threshold=1.):
    # load all data for a given recording

    conditions_alignment = {'onset': ['RandReg', 'RegRand', 'Rand',
                                      'Reg', 'Const', 'Step'],
                            'transition': ['RandReg', 'RegRand', 'Step']}

    # load different quantities
    ts, pupil_size = load_pupil_data(rec_path)

    stim_events, conditions = load_stim_events(rec_path)
    print("conditions:", conditions)

    t, backward, forward, speed = load_cylinder_running_data(rec_path)

    # use list comprehensions to create arrays of stim_event
    transition_times = np.asarray([ev['Transition']for ev in stim_events])
    stim_time = np.asarray([ev['Timestamp'] for ev in stim_events])
    cond_arr = np.asarray([ev['Stim']for ev in stim_events])

    # create function for interpolation for pupil size
    f1 = interpolate.interp1d(ts, pupil_size,
                              kind='linear',
                              bounds_error=False,
                              fill_value=np.NaN)

    # interpolation function for speed data
    f2 = interpolate.interp1d(t, speed,
                              kind='linear',
                              bounds_error=False,
                              fill_value=0)

    T = max([ev['Duration'] for ev in stim_events])
    n_gridpoints = int(round((t_pre + T + t_post)/dt + .5))
    t_grid = -t_pre + np.arange(n_gridpoints) * dt
    n_pre = int(round(t_pre/dt))

    t_norm = min(t_norm, t_pre)  # norm. window cannot be longer than t_pre
    n_norm = int(round(t_norm/dt))

    results = {}

    for i, cond in enumerate(conditions_alignment[alignment]):

        print(i, cond)
        v = cond_arr == cond

        # --- align to stimulus onset ---
        n_trials = np.sum(v)

        if n_trials > 0:

            X = np.zeros((n_trials, n_gridpoints))
            S = np.zeros((n_trials, n_gridpoints))

            tt = stim_time[v]
            if alignment == 'transition':
                tt += transition_times[v]

            for j, t0 in enumerate(tt):  # looping over trials

                X[j, :] = f1(t0 + t_grid)

                if normalize:
                    norm_trial = np.mean(X[j, (n_pre-n_norm):n_pre])
                    X[j, :] /= norm_trial

                S[j, :] = f2(t0 + t_grid)

            max_speed = np.max(S, axis=1)
            is_running = max_speed > running_threshold

            print('run=', np.sum(is_running == 1))
            print('still=', np.sum(is_running == 0))

            results[cond] = {'pupil': X,
                             'speed': S,
                             'max_speed': max_speed,
                             'is_running': is_running,
                             't_grid': t_grid}

    return results


def compute_mean_std_nan(trials):

    if np.any(trials):

        n_gridpoints = trials.shape[1]
        mx = np.zeros((n_gridpoints,))
        sx = np.zeros((n_gridpoints,))
        valid = np.zeros((n_gridpoints,))

        for k in range(n_gridpoints):
            vv = ~np.isnan(trials[:, k])
            valid[k] = np.sum(vv)
            if valid[k] < 5:
                trials[vv, k] = np.NaN
            mx[k] = np.nanmean(trials[vv, k])
            sx[k] = np.nanstd(trials[vv, k])

    else:
        mx = None
        sx = None
        valid = None

    return mx, sx, valid


def get_color_condition(cond):

    colors = {'Rand': 3*[.4],
              'Const': 3*[.7],
              'RegRand': '#1f77b4',
              'Step': '#ff7f0e',
              'RandReg': '#bcbd22',
              'Reg': '#e377c2'}

    return colors[cond]


def set_font_axes(ax, add_size=0, size_ticks=6, size_labels=8,
                  size_text=8, size_title=8, family='Arial'):

    if size_title is not None:
        ax.title.set_fontsize(size_title + add_size)

    if size_ticks is not None:
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=size_ticks + add_size)

    if size_labels is not None:

        ax.xaxis.label.set_fontsize(size_labels + add_size)
        ax.xaxis.label.set_fontname(family)

        ax.yaxis.label.set_fontsize(size_labels + add_size)
        ax.yaxis.label.set_fontname(family)

        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(size_labels + add_size)
            ax.zaxis.label.set_fontname(family)

    if size_text is not None:
        for at in ax.texts:
            at.set_fontsize(size_text + add_size)
            at.set_fontname(family)


def adjust_axes(ax, tick_length=True, tick_direction=True,
                spine_width=0.5, pad=2):

    if tick_length:
        ax.tick_params(axis='both', which='major', length=2)

    if tick_direction:
        ax.tick_params(axis='both', which='both', direction='out')

    if pad is not None:
        ax.tick_params(axis='both', which='both', pad=pad)

    for s in ax.spines:
        spine = ax.spines[s]
        if spine.get_visible():
            spine.set_linewidth(spine_width)


def simple_xy_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def simple_twinx_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')


@click.command(name='recording')
@click.argument('rec_path', type=click.Path(exists=True))
def cli_recording(rec_path=None):

    # load data
    ts, pupil_size = load_pupil_data(rec_path)

    stim_events, conditions = load_stim_events(rec_path)
    print("conditions:", conditions)

    t, backward, forward, speed = load_cylinder_running_data(rec_path)

    # use list comprehensions to create arrays of stim_event
    transition_times = np.asarray([ev['Transition']for ev in stim_events])
    stim_time = np.asarray([ev['Timestamp'] for ev in stim_events])
    cond_arr = np.asarray([ev['Stim']for ev in stim_events])
#    durs = np.asarray([ev['Duration']for ev in stim_events])

    # create function for interpolation for pupil size
    f1 = interpolate.interp1d(ts, pupil_size,
                              kind='linear',
                              bounds_error=False,
                              fill_value=np.NaN)

    # interpolation function for speed data
    f2 = interpolate.interp1d(t, speed,
                              kind='linear',
                              bounds_error=False,
                              fill_value=0)

    t_pre = 2.
    t_post = 2.
    dt = .05
    normalize = True
    running_threshold = 1.
    T = max([ev['Duration'] for ev in stim_events])

    n_gridpoints = int(round((t_pre + T + t_post)/dt + .5))
    t_grid = -t_pre + np.arange(n_gridpoints) * dt
    n_pre = int(round(t_pre/dt))

    n_cond = len(conditions)
    fig, axarr = plt.subplots(nrows=4, ncols=n_cond,
                              sharex=True, sharey=True)

    norm_cond_trials = {}  # use same normalizer for onset/transition alignment

    for i, cond in enumerate(conditions):  # looping over the 6 conditions

        print(i, cond)
        v = cond_arr == cond

        # --- align to stimulus onset ---
        n_trials = np.sum(v)
        X = np.zeros((n_trials, n_gridpoints))
        S = np.zeros((n_trials, n_gridpoints))
        norm_trials = np.zeros((n_trials,))  # normalizer for each trial

        for j, t0 in enumerate(stim_time[v]):  # looping over trials

            X[j, :] = f1(t0 + t_grid)
            S[j, :] = f2(t0 + t_grid)

            # mean pupil size of the pre-event time.
            norm_trials[j] = np.mean(X[j, :n_pre])


        if normalize:
            # this approach is using broadcasting; for details see
            # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
            X = (X.T / norm_trials).T

        norm_cond_trials[cond] = norm_trials

        is_running = np.max(S, axis=1) > running_threshold
        run_trials = X[is_running, :]
        norun_trials = X[~is_running, :]

        print('run=', run_trials.shape[0])
        print('still=', norun_trials.shape[0])

        # do things in a loop
        for ii, (what, trials) in enumerate([('Running', run_trials),
                                            ('Still', norun_trials)]):

            ax = axarr[ii, i]

            # only carry through if array is not empty
            if np.any(trials):

                mx = np.zeros((n_gridpoints,))
                sx = np.zeros((n_gridpoints,))
                valid = np.zeros((n_gridpoints,))

                for k in range(n_gridpoints):
                    vv = ~np.isnan(trials[:, k])
                    valid[k] = np.sum(vv)
                    if valid[k] < 2:
                        trials[vv, k] = np.NaN
                    mx[k] = np.nanmean(trials[vv, k])
                    sx[k] = np.nanstd(trials[vv, k])

                # plot data - to onset- running
                ax.set_title('{} {}'.format(what, cond))
                ax.set_xlabel('Time (s)', labelpad=2)
                ax.set_ylabel('Pupil size (au)', labelpad=2)

                ax.plot(t_grid, trials.T, '-', color=3*[.5], lw=.5)
                ax.plot(t_grid, mx, '-', color='r', lw=1)
                ax.fill_between(t_grid, mx - sx, mx + sx,
                                color=3*[.5], alpha=.5)
                ax.axvline(x=0, linestyle='--', color='r', lw=1)
                if cond in ['RandReg', 'RegRand', 'Step']:
                    # note that -1 means no transition time
                    for t0 in transition_times[v]:
                        ax.axvline(t0,
                                   ls='--',
                                   color='k',
                                   lw=0.5)
                #plot twinx
                ax2 = ax.twinx()
                ax2.plot(t_grid, valid.T, '-', color='b', lw=1)
                ax2.set_ylim(0,100)
                ax2.set_ylabel('Valid trials')
            else:
                ax.axis('off')

        # --- align to transition (if applicable) ---
        tt_mean = np.mean(transition_times[v])
        if cond in ['RandReg', 'RegRand', 'Step']:

            Y = np.zeros((n_trials, n_gridpoints))
            S = np.zeros((n_trials, n_gridpoints))

            for j, (t0, tt) in enumerate(zip(stim_time[v],
                                             transition_times[v])):
                # shift by difference between actual and expected (mean)
                # transition time
                Y[j, :] = f1(t0 + tt - tt_mean + t_grid)
                S[j, :] = f2(t0 + tt - tt_mean + t_grid)

            if normalize:
                norm_trials = norm_cond_trials[cond]
                Y = (Y.T / norm_trials).T

            is_running = np.max(S, axis=1) > running_threshold
            run_trials = Y[is_running, :]
            norun_trials = Y[~is_running, :]

            print('run=', run_trials.shape[0])
            print('still=', norun_trials.shape[0])

            for ii, (what, trials) in enumerate([('Running', run_trials),
                                                 ('Still', norun_trials)]):
                ax = axarr[2+ii, i]

                if np.any(trials):

                    my = np.zeros((n_gridpoints,))
                    sy = np.zeros((n_gridpoints,))
                    valid = np.zeros((n_gridpoints,))

                    for k in range(n_gridpoints):
                        vv = ~np.isnan(trials[:, k])
                        valid[k] = np.sum(vv)
                        if valid[k] < 2:
                            trials[vv, k] = np.NaN
                        my[k] = np.nanmean(trials[vv, k])
                        sy[k] = np.nanstd(trials[vv, k])

                    # plot data - to transition- running
                    ax.set_title('{} {}'.format(what, cond))
                    ax.set_xlabel('Time (s)', labelpad=2)
                    ax.set_ylabel('Pupil size (au)', labelpad=2)
                    ax.plot(t_grid, trials.T, '-', color=3 * [.5], lw=.5)
                    ax.plot(t_grid, my, '-', color='r', lw=1)
                    ax.fill_between(t_grid, my - sy, my + sy,
                                    color=3 * [.5], alpha=.5)
                    ax.axvline(x=0, linestyle='--', color='r', lw=1)
                    # show stimulus onset times
                    for t0 in transition_times[v]:
                        ax.axvline(t0 - tt_mean,
                                   ls='--',
                                   color='k',
                                   lw=0.5)
                    ax.axvline(tt_mean,
                               linestyle='--',
                               color='r',
                               lw=1)
                    # plot twinx
                    ax2 = ax.twinx()
                    ax2.plot(t_grid, valid, lw=0.2, color='b')
                    set_font_axes(ax2)
                    ax2.set_ylim(40, 110)
                    ax2.set_yticks([50, 75, 100])
                    ax2.tick_params(axis='y', labelcolor='b')
                    ax2.set_ylabel('Valid Trials', color='b')
                else:
                    ax = axarr[2+ii, i]
                    ax.axis('off')
        else:
            ax = axarr[2, i]
            ax.axis('off')
            ax = axarr[3, i]
            ax.axis('off')

    # prettify figure a bit
    for ax in fig.axes:
        set_font_axes(ax)
        simple_xy_axes(ax)
        adjust_axes(ax)
        ax.set_ylim(.75, 1.4)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        # set_font_axes(ax2)
        # simple_xy_axes(ax2)
        # adjust_axes(ax2)
        # ax2.set_ylim(0, 50)
        # ax2.yaxis.show_ticks()

    fig.set_size_inches(7.1, 3.25)
    fig.tight_layout(w_pad=.25)

    rec_name = op.split(rec_path)[-1]
    fig_file = op.join(op.split(__file__)[0], 'randreg_{}'.format(rec_name))
    for ff in ['png', 'pdf', 'svg']:
        fig.savefig(fig_file + '.' + ff,
                    format=ff,
                    dpi=300)

    plt.show()


@click.command(name='summarize')
@click.argument('db_path', type=click.Path(exists=True))
def cli_summarize(db_path=None):

    # mouse subdirectories
    mice = sorted([d for d in os.listdir(db_path) if d.startswith('M102')])
    print(mice)

    conditions = ['RandReg', 'RegRand', 'Rand',
                  'Reg', 'Const', 'Step']

    running_threshold = 1.
    t_pre = 4.
    t_post = 4.
    dt = .05

    mouse = 'M10205'
    recordings = ['M10205/2018_05_24/reg2/2018-05-24_12-34-34_ChaitStimulus',
                  'M10205/2018_05_25/reg2/2018-05-25-_14-35-22_ChaitStimulus',
                  'M10205/2018_05_26/reg2/2018-05-26_11-32-04_ChaitStimulus',
                  'M10205/2018_05_28/reg2/2018-05-28_14-59-42_ChaitStimulus',
                  'M10205/2018_05_29/reg2/2018-05-29_14-43-26_ChaitStimulus',
                  'M10205/2018_05_30/reg2/2018-05-30_17-40-39_ChaitStimulus',
                  'M10205/2018_05_31/reg2/2018-05-31_13-49-06_ChaitStimulus',
                  'M10205/2018_06_01/reg2/2018-06-01_15-04-27_ChaitStimulus',
                  'M10205/2018_06_11/reg2/2018-06-11_12-34-51_ChaitStimulus']

        #mouse_path = op.join(db_path, mouse)
        # for root, dirs, files in os.walk(mouse_path):
        #     if 'ChaitStimulus' in root and \
        #             'rpi_camera_2_pupil_data.npz' in files:
        #         # this includes all recordings with pupil data; adapt filtering
        #         # or use hard-coded recordings for final analysis?
        #         rec_paths.append(root)

    fig, axarr = plt.subplots(nrows=4, ncols=len(conditions))

    for i, alignment in enumerate(['onset', 'transition']):

        results_cond = {}
        for recording in recordings:

            rec_path = op.join(db_path, recording)
            try:
                res = load_data_recording(rec_path,
                                          alignment=alignment,
                                          t_pre=t_pre,
                                          t_post=t_post,
                                          dt=dt,
                                          normalize=True,
                                          running_threshold=running_threshold)

                for k in res:
                    if k not in results_cond:
                        results_cond[k] = []
                    results_cond[k].append(res[k])
            except BaseException:
                traceback.print_exc()

        # do things in a loop
        for j, cond in enumerate(conditions):

            for k, what in enumerate(['Still', 'Running']):

                row = i*2+k
                col = conditions.index(cond)
                ax = axarr[row, col]

                if cond in results_cond:

                    rr = results_cond[cond]

                    # due to rounding errors one recording had a different
                    # number of grid points; simply ignore last few time
                    # steps
                    ng = min([r['pupil'].shape[1] for r in rr])
                    pupil = np.concatenate([r['pupil'][:, :ng]
                                            for r in rr],
                                           axis=0)
                    speed = np.concatenate([r['max_speed'] for r in rr],
                                           axis=0)
                    t_grid = rr[0]['t_grid']

                    # split into still/running trials
                    if what == 'Still':
                        v = speed < running_threshold
                    else:
                        v = speed >= running_threshold

                    trials = pupil[v, :]
                    mx, sx, valid = compute_mean_std_nan(trials)

                    # plot data - to onset- running
                    ax.set_title('{} {}\n({})'.format(
                            what, cond, alignment))

                    ax.plot(t_grid, trials.T, '-', color=3*[.5],
                            lw=.25, alpha=.25)
                    ax.axvline(x=0, linestyle='--', color=3*[.2], lw=1)
                    ax.fill_between(t_grid, mx - sx, mx + sx,
                                    color=3*[.5], alpha=.5)
                    ax.plot(t_grid, mx, '-', color='#d62728', lw=1)

                    if row == 3:
                        ax.set_xlabel('Time (s)', labelpad=2)
                    if col == 0:
                        ax.set_ylabel('Norm. pupil size', labelpad=2)

                    set_font_axes(ax)
                    simple_xy_axes(ax)
                    adjust_axes(ax)
                    ax.set_ylim(.7, 1.3)
                    ax.set_yticks([0.7, 1, 1.3])
                    ax.set_yticklabels(['0.7', '1', '1.3'])
                    ax.set_xlim(-4., 4.)
                    ax.set_xticks([-4, 0, 4])

                else:
                    ax.axis('off')

#        fig.tight_layout()
    fig.subplots_adjust(left=.08, right=.94, bottom=.1, top=.925,
                        wspace=.5, hspace=.5)
        #fig.suptitle(mouse, family='Arial', fontsize=10)

    plt.show()


@click.group()
def cli():
    pass


cli.add_command(cli_recording)
cli.add_command(cli_summarize)


if __name__ == '__main__':
    cli()