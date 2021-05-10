#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
note: STRF estimation requires the lnpy package: https://github.com/arnefmeyer/lnpy

Created on Mon Jul  9 14:43:11 2018

@author: arne, marios
"""


from __future__ import print_function

import os
import os.path as op
import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import six
import traceback
import pdb

from stimcontrol.io.database import NeoIO
from stimcontrol.util import makedirs_save
import stimcontrol.util as scu
import helpers

def plot_waveforms(ax, unit):

    wf = unit.annotations['waveform']
    Y = wf['mean']

    y_max = np.max(np.abs(Y))

    n_channels = Y.shape[1]
    offset = 0
    for i in range(n_channels):

        y = Y[:, i]
        x = np.arange(y.shape[0]) / float(wf['samplerate'])
        ax.plot(x, y + i*y_max, '-', color=3*[.2], linewidth=2)

        offset += y.shape[0] + 5

    ax.axis('off')


def get_clustergroup_color(name):

    c = 3 * [.25]
    if name.lower() == 'mua':
        c = scu.NICE_COLORS['tomato']
    elif name.lower() == 'good':
        c = scu.NICE_COLORS['forest green']
    elif name.lower() == 'noise':
        c = 3 * [.75]
    elif name.lower() == 'unsorted':
        c = scu.NICE_COLORS['deep sky blue']

    return c
def get_y_lim(seg, unit,
              binwidth=.1,
                 pre=0,
                 post=0,
                 **kwargs):

    index = [st.unit for st in seg.spiketrains].index(unit)
    train = seg.spiketrains[index].magnitude
    events = seg.events

    duration = np.max([ev.annotations['Duration'] for ev in events])
    n_bins = int(np.ceil((pre + duration + post) / binwidth))
    edges = -pre + np.arange(n_bins + 1) * binwidth

    spike_times = []
    for ev in events:
        t0 = ev.time
        v = np.logical_and(train >= t0 - pre,
                           train <= t0 + post)
        spike_times.append(train[v] - t0)

    psth = np.histogram(np.concatenate(spike_times),
                        bins=edges)[0] / binwidth / len(events)
    return np.max(psth)

def compute_psth(seg, unit,
                 binwidth=.1,
                 pre=0,
                 post=0,
                 ax=None,
                 ymax = None,
                 **kwargs):

    index = [st.unit for st in seg.spiketrains].index(unit)
    train = seg.spiketrains[index].magnitude
    events = seg.events

    duration = np.max([ev.annotations['Duration'] for ev in events])
    n_bins = int(np.ceil((pre + duration + post) / binwidth))
    edges = -pre + np.arange(n_bins + 1) * binwidth

    spike_times = []
    for ev in events:
        t0 = ev.time
        v = np.logical_and(train >= t0 - pre,
                           train <= t0 + post)
        spike_times.append(train[v] - t0)

    psth = np.histogram(np.concatenate(spike_times),
                        bins=edges)[0] / binwidth / len(events)

    if ax is not None:

        ax.bar(edges[:-1], psth, width=binwidth,
               color=3*[.5],
               edgecolor='none',
               linewidth=0,
               align='edge')
        ax.axvspan(0, duration,
                   color=[0, 0.5, 1],
                   alpha=.5,
                   edgecolor='none',
                   lw=0,
                   zorder=0)
        ax.set_ylim(0, ymax)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Spikes/s')

    return psth.max()

def get_tone_r_max(segment, unit):
    index = [st.unit for st in segment.spiketrains].index(unit)
    train = segment.spiketrains[index].magnitude
    events = segment.events

    tone_len = [ev.annotations['Duration'] for ev in events]
    tone_len = np.unique(tone_len)[0]
    seq_levels = [ev.annotations['Level'] for ev in events]
    seq_freqs = [ev.annotations['Frequency'] for ev in events]

    levels = np.unique(seq_levels)
    freqs = np.unique(seq_freqs)

    R = np.zeros((len(levels), len(freqs)), dtype=np.float64)
    trial_cnt = np.zeros_like(R)
    for i, ll in enumerate(levels):

        for j, ff in enumerate(freqs):

            for ii, ev in enumerate(events):

                if seq_levels[ii] == ll and seq_freqs[ii] == ff:
                    t0 = ev.time
                    valid = np.logical_and(train >= t0,
                                           train < t0 + tone_len)
                    R[i, j] += np.sum(valid)
                    trial_cnt[i, j] += 1

    R /= (trial_cnt * tone_len)

    return np.max(R)

def analyze_tone_sequence(segment, unit,
                          ax=None,
                          R_max = None,
                          **kwargs):

    index = [st.unit for st in segment.spiketrains].index(unit)
    train = segment.spiketrains[index].magnitude
    events = segment.events

    tone_len = [ev.annotations['Duration'] for ev in events]
    tone_len = np.unique(tone_len)[0]
    seq_levels = [ev.annotations['Level'] for ev in events]
    seq_freqs = [ev.annotations['Frequency'] for ev in events]

    levels = np.unique(seq_levels)
    freqs = np.unique(seq_freqs)

    R = np.zeros((len(levels), len(freqs)), dtype=np.float64)
    trial_cnt = np.zeros_like(R)
    for i, ll in enumerate(levels):

        for j, ff in enumerate(freqs):

            for ii, ev in enumerate(events):

                if seq_levels[ii] == ll and seq_freqs[ii] == ff:

                    t0 = ev.time
                    valid = np.logical_and(train >= t0,
                                           train < t0 + tone_len)
                    R[i, j] += np.sum(valid)
                    trial_cnt[i, j] += 1

    R /= (trial_cnt * tone_len)

#    from scipy.ndimage.filters import gaussian_filter
#    R = gaussian_filter(R, .75)

    if ax is not None:

        ax.imshow(R, vmin=0, vmax=R_max, aspect='auto',
                  interpolation='nearest', origin='lower')

        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Level (dB)')

        ax.set_xticks(range(0, len(freqs), 4))
        fl = ['%0.1f' % x for x in freqs[::4] / 1000.]
        ax.set_xticklabels(fl)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels(np.asarray(levels, dtype=int))

    return R, freqs, levels


def compute_strf(segment, unit,
                 max_lag=.24,
                 ax=None,
                 method='Ridge',
                 **kwargs):

    from lnpy.linear import Ridge, ASD

    index = [st.unit for st in segment.spiketrains].index(unit)
    spike_train = segment.spiketrains[index].magnitude
    stim_events = segment.events

    rec_path = segment.file_origin
    dd = helpers.load_drc_data(rec_path,
                               max_lag=max_lag)

    rr = helpers.bin_spike_train(spike_train, stim_events,
                                 max_lag=max_lag,
                                 trial_duration=dd['trial_duration'])

    # fit model parameters using time-lagged data (see load function)
    XX = dd['XX']
    yy = rr['yy']
    rf_size = dd['rf_size']

    if method.title() in ['Ridge', 'Linear']:
        model = Ridge(verbose=1,
                      maxiter=1e3,
                      tolerance=1e-3,
                      solver='iter',
                      alpha0=1.)
    elif method.title() in ['ASD', 'Smooth']:
        model = ASD(D=rf_size,
                    fit_bias=True,
                    verbose=True,
                    maxiter=100,
                    solver='iter',
                    smooth_min=.5,
                    init_params=[7, 4, 4],
                    tolerance=0.1)

    model.fit(XX, yy)

    if ax is not None:

        model.show(shape=rf_size,
                   dt=dd['dt'],
                   ax=ax,
                   show_now=False,
                   frequencies=dd['f_center'],
                   colorbar = True)
        t = ax.set_title(method.title())
        t.set_y(.95)


def analyze_recordings(block, units, units_per_page=1, **kwargs):

    func_mappings = {}

    func_mappings['NoiseBurst'] = \
        [('PSTH', compute_psth)]

    func_mappings['ToneSequence'] = \
        [('PSTH', compute_psth),
         ('Tuning', analyze_tone_sequence)]

    func_mappings['FMSweep'] = \
        [('PSTH', compute_psth)]

    func_mappings['DRC'] = \
        [('STRF', compute_strf)]

    func_list = []
    for seg in block.segments:

        for name, funcs in six.iteritems(func_mappings):

            if name in seg.name:

                for title, func in funcs:
                    func_list.append((title, func, seg))

    n_rows = len(func_list) + 1

    fig, axarr = plt.subplots(nrows=n_rows, ncols=units_per_page)

    seg_labels = []
    for i, unit in enumerate(units):

        row_cnt = 0
        ax = axarr[row_cnt, i]
        try:
            plot_waveforms(ax, unit)
            seg_labels.append(('Waveforms', '', ''))
        except:
            print("could not plot waveforms")

        ax.set_title('Chan{:d} Unit{:d}'.format(
            unit.annotations['channel_index'],
            unit.annotations['unit_index']),
            color=get_clustergroup_color(unit.description)
            )

        row_cnt += 1

        for title, func, seg in func_list:

            ax = axarr[row_cnt, i]
            row_cnt += 1

            try:
                func(seg, unit, ax=ax, **kwargs)
            except KeyboardInterrupt:
                return
            except BaseException:
                traceback.print_exc()

            seg_labels.append((seg.name[len(seg.rec_datetime)+1:],
                               seg.rec_datetime,
                               title))
    for ax in axarr.flat:
        scu.set_font_axes(ax, add_size=-1)
        scu.simple_xy_axes(ax)

    if len(units) < units_per_page:
        for ax in axarr[:, len(units):].ravel():
            ax.axis('off')

    fig.set_size_inches(8, 1.75*n_rows)  # 1.5 * n_rows)
    rect = (0.15, 0.025, 1, 1)
    fig.tight_layout(pad=.1, h_pad=.075, w_pad=.025, rect=rect)

    try:
        for j, ax in enumerate(axarr[:, 0]):

            y0 = ax.get_position().y0
            y1 = ax.get_position().y1

            fig.text(0.01, y0 + .4*(y1 - y0), '{}\n{}\n -- {}'.format(
                    seg_labels[j][0],
                    seg_labels[j][1],
                    seg_labels[j][2]),
                     fontsize=5, family='Arial', va='center')
    except:
        traceback.print_exc()

    return fig

def analyze_recordings_temp(block, unit, units_per_page=1, **kwargs):

    fig, axarr = plt.subplots(nrows=6, ncols=4)
    seg_labels = []
    for ax in axarr[0,1:]:
        ax.axis('off')

    ax = axarr[0, 0]
    try:
        plot_waveforms(ax, unit)
        seg_labels.append(('Waveforms', '', ''))
    except:
        print("could not plot waveforms")

    ax.set_title('Chan{:d} Unit{:d}'.format(
        unit.annotations['channel_index'],
        unit.annotations['unit_index']),
        color=get_clustergroup_color(unit.description)
    )

    noise_y_max = []
    tone_y_max = []
    tone_r_max = []
    drc_y_max = []
    #loop over the segments to find the maximum for each type:
    for i, seg in enumerate(block.segments[:16]):
        if 'NoiseBurst' in seg.name:
            noise_y_max.append(get_y_lim(seg, unit))
        if 'ToneSequence' in seg.name:
            tone_y_max.append(get_y_lim(seg, unit))
            tone_r_max.append(get_tone_r_max(seg, unit))
        if 'DRC' in seg.name:
            drc_y_max.append(get_y_lim(seg, unit))

    row = 1
    for i, seg in enumerate(block.segments[:4]):

        seg_labels.append((seg.name[len(seg.rec_datetime) + 1:],
                           seg.rec_datetime))
        if 'NoiseBurst' in seg.name:
            #plot PSTH
            compute_psth(seg, unit, ax = axarr[row,0], y_max = max(noise_y_max), **kwargs)
            row +=1
        if 'ToneSequence' in seg.name:
            # plot PSTH
            compute_psth(seg, unit, ax=axarr[row, 0], y_max = max(tone_y_max), **kwargs)
            row += 1
            # plot tuning
            analyze_tone_sequence(seg, unit, ax=axarr[row, 0], R_max = get_tone_r_max(seg, unit),  **kwargs)
            row += 1
        if 'DRC' in seg.name:
            compute_strf(seg, unit, ax= axarr[row, 0], **kwargs)
    row = 1
    for i, seg in enumerate(block.segments[4:8]):

        seg_labels.append((seg.name[len(seg.rec_datetime) + 1:],
                           seg.rec_datetime))
        if 'NoiseBurst' in seg.name:
            #plot PSTH
            compute_psth(seg, unit, ax = axarr[row,1],y_max = max(noise_y_max), **kwargs)
            row +=1
        if 'ToneSequence' in seg.name:
            # plot PSTH
            compute_psth(seg, unit, ax=axarr[row, 1], y_max = max(tone_y_max), **kwargs)
            row += 1
            # plot tuning
            analyze_tone_sequence(seg, unit, ax=axarr[row, 1],R_max = get_tone_r_max(seg, unit),  **kwargs)
            row += 1
        if 'DRC' in seg.name:
            compute_strf(seg, unit, ax= axarr[row, 1], **kwargs)
    row = 1
    for i, seg in enumerate(block.segments[8:12]):
        seg_labels.append((seg.name[len(seg.rec_datetime) + 1:],
                           seg.rec_datetime))
        if 'NoiseBurst' in seg.name:
            #plot PSTH
            compute_psth(seg, unit, ax = axarr[row,2],y_max = max(noise_y_max), **kwargs)
            row +=1
        if 'ToneSequence' in seg.name:
            # plot PSTH
            compute_psth(seg, unit, ax=axarr[row, 2], y_max = max(tone_y_max),**kwargs)
            row += 1
            # plot tuning
            analyze_tone_sequence(seg, unit, ax=axarr[row, 2], **kwargs)
            row += 1
        if 'DRC' in seg.name:
            compute_strf(seg, unit, ax = axarr[row, 2],  **kwargs)
    row = 1

    for i, seg in enumerate(block.segments[12:16]):
        seg_labels.append((seg.name[len(seg.rec_datetime) + 1:],
                           seg.rec_datetime))
        if 'NoiseBurst' in seg.name:
            #plot PSTH
            compute_psth(seg, unit, ax = axarr[row,3],y_max = max(noise_y_max), **kwargs)
            row +=1
        if 'ToneSequence' in seg.name:
            # plot PSTH
            compute_psth(seg, unit, ax=axarr[row, 3], y_max = max(tone_y_max),**kwargs)
            row += 1
            # plot tuning
            analyze_tone_sequence(seg, unit, ax=axarr[row, 3], R_max = get_tone_r_max(seg, unit), **kwargs)
            row += 1
        if 'DRC' in seg.name:
            compute_strf(seg, unit, ax= axarr[row, 3], **kwargs)

    for ax in axarr.flat:
        scu.set_font_axes(ax, add_size=-1)
        scu.simple_xy_axes(ax)

    fig.set_size_inches(8, 1.75 * 6)  # 1.5 * n_rows)
    rect = (0.15, 0.025, 1, 1)
    fig.tight_layout(pad=.1, h_pad=.075, w_pad=.025, rect=rect)
    #pdb.set_trace()
    try:
        for j, label in enumerate(seg_labels[:5]):
            # y0 = ax.get_position().y0
            # y1 = ax.get_position().y1

            fig.text(0.01, 0.95 - .22 * j, '{}'.format(
                label[0]), fontsize=5, family='Arial', va='center')
    except:
        traceback.print_exc()
    axarr[1,2].set_title('Anaesthetised recordings')
    return fig


#  arne's functions for normal recordings
# @click.command()
# @click.argument('session_path', type=click.Path(exists=True))
# @click.option('--output', '-o', type=click.Path(exists=False))
# @click.option('--pre', '-p', default=.5)
# @click.option('--post', '-P', default=.5)
# @click.option('--binwidth', '-b', default=.025)
# @click.option('--show', '-s', is_flag=True)
# def cli(session_path=None,
#         output=None,
#         units_per_page=1,
#         show=False,
#         **kwargs):
#
#     recordings = helpers.get_recordings(session_path)
#     reader = NeoIO(session_path)
#     block = reader.read_block(recordings=recordings,
#                               dtypes=['spikes'],
#                               cluster_groups=['Good', 'MUA'])
#     units = block.list_units
#     n_units = len(units)
#
#     print("    segments:", len(block.segments))
#     print("    units:", len(units))
#
#     if n_units > 0:
#
#         # create output directory
#         if output is None:
#             output = op.join(session_path, 'analysis', 'tuning', session_path.split(op.sep)[-3])
#
#         makedirs_save(output)
#         print("  saving results to:", output)
#
#         file_name = helpers.create_filename_from_session_path(
#             session_path)
#         pdf_path = op.join(output, file_name)
#         pdf = PdfPages(pdf_path + '.pdf')
#
#         n_pages = int(np.ceil(n_units / float(units_per_page)))
#         for i in range(n_pages):
#
#             print("  pdf page: {}/{}".format(i+1, n_pages))
#
#             i1 = i*units_per_page
#             i2 = min((i+1)*units_per_page, n_units)
#             fig = analyze_recordings_temp(block, units[i1:i2],
#                                      units_per_page=units_per_page,
#                                      **kwargs)
#
#             if output is not None:
#                 pdf.savefig(fig)
#
#             if not show:
#                 plt.close(fig)
#
#         if output is not None:
#             pdf.close()
#
#
# if __name__ == '__main__':
#     cli()


#marios's function for anaesthetised recordings
@click.command()
@click.argument('session_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--pre', '-p', default=.5)
@click.option('--post', '-P', default=.5)
@click.option('--binwidth', '-b', default=.025)
@click.option('--show', '-s', is_flag=True)
def cli(session_path=None,
        output=None,
        units_per_page=1,
        show=False,
        **kwargs):

    recordings = helpers.get_recordings(session_path)
    reader = NeoIO(session_path)
    block = reader.read_block(recordings=recordings,
                              dtypes=['spikes'],
                              cluster_groups=['Good', 'MUA'])
    units = block.list_units
    n_units = len(units)

    print("    segments:", len(block.segments))
    print("    units:", len(units))

    if n_units > 0:

        # create output directory
        if output is None:
            output = op.join(session_path, 'analysis', 'tuning', session_path.split(op.sep)[-3])

        makedirs_save(output)
        print("  saving results to:", output)

        file_name = helpers.create_filename_from_session_path(
            session_path)
        pdf_path = op.join(output, file_name)
        pdf = PdfPages(pdf_path + '.pdf')

        n_pages = int(np.ceil(n_units / float(units_per_page)))
        for i, unit in enumerate(units):

            print("  pdf page: {}/{}".format(i+1, n_pages))

            fig = analyze_recordings_temp(block, unit,
                                     units_per_page=units_per_page,
                                     **kwargs)

            if output is not None:
                pdf.savefig(fig)

            if not show:
                plt.close(fig)

        if output is not None:
            pdf.close()


if __name__ == '__main__':
    cli()
