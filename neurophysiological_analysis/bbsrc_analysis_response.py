#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:23:31 20190 

@author: marios
"""

import matplotlib
import numpy as np
import math
import os
import click
import os.path as op
from scipy import mean
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from stimcontrol.io.database import NeoIO
import stimcontrol.util as scu
from scipy.stats import wilcoxon
import helpers
from collections import Counter
import pdb
from matplotlib_venn import venn3

def cm2inch(value):
    return value/2.54

def get_recordings(session_path, patterns=['_NoiseBurst',
                                           '_FMSweep',
                                           '_ToneSequence',
                                           '_DRC']):

    rec_dirs = []
    for d in os.listdir(session_path):
        if len([d for p in patterns if p in d]) > 0:
            rec_dirs.append(d)

    return list(sorted(set(rec_dirs)))

def get_drc_segments(session_path, patterns=['_DRC']):

    rec_dirs = []
    for d in os.listdir(session_path):
        if len([d for p in patterns if p in d]) > 0:
            rec_dirs.append(d)

    return list(sorted(set(rec_dirs)))


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


def compute_psth(seg, unit,
                 binwidth=0.1,
                 pre=0,
                 post=0,
                 ax=None,
                 **kwargs):

    index = [st.unit for st in seg.spiketrains].index(unit)
    train = seg.spiketrains[index].magnitude *1000

    events = seg.events

    duration = np.max([ev.annotations['Duration']*1000 for ev in events])
    n_bins = int(np.ceil((pre + post) / binwidth))
    edges = -pre + np.arange(n_bins + 1) * binwidth

    spike_times = []
    for ev in events:
        t0 = ev.time *1000
        v = np.logical_and(train >= t0 - pre,
                           train <= t0 + post)
        spike_times.append(train[v] - t0)

    psth = np.histogram(np.concatenate(spike_times),
                        bins=edges)[0] # / binwidth / len(events)
    #pdb.set_trace()
    # if ax is not None:
    #
    #     ax.bar(edges[:-1], psth, width=binwidth,
    #            color=3*[.5],
    #            edgecolor='none',
    #            linewidth=0,
    #            align='edge')
    #     ax.axvspan(0, duration,
    #                color=[0, 0.5, 1],
    #                alpha=.5,
    #                lw=0,
    #                zorder=0)
    #
    #     ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    #     ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    return psth, edges, len(events), duration


def first_spike_hist(seg, unit,
                     binwidth=.1,
                     tmax=.06,
                     ax=None,
                     **kwargs):
    
    index = [st.unit for st in seg.spiketrains].index(unit)
    train = seg.spiketrains[index].magnitude
    events = seg.events

    duration = np.max([ev.annotations['Duration'] for ev in events])
    n_bins = int(np.ceil((tmax) / binwidth))
    edges = np.arange(n_bins + 1) * binwidth
    
    first_spikes = []
    for ev in events:
        t0 = ev.time
        first = np.logical_and(train >= t0,
                               train <= t0 + tmax)
        if len(train[first] > 0):
            first_spikes.append(train[first][0] - t0)
        
    fspike = np.histogram(first_spikes,
                          bins=edges)[0] / binwidth / len(events)
    
    if ax is not None:

        ax.bar(edges[:-1], fspike, width=binwidth,
               color=3*[.5],
               edgecolor='none',
               linewidth=0,
               align='edge')
        ax.axvspan(0, duration,
                   color=[0, 0.5, 1],
                   alpha=.5,
                   lw=0,
                   zorder=0)
    
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    return fspike.max()

    
def plot_fs_intensity(seg, unit,
                      tmax=.06,
                      ax=None,
                      **kwargs):
    
    index = [st.unit for st in seg.spiketrains].index(unit)
    train = seg.spiketrains[index].magnitude
    events = seg.events

    first_spikes=[]
    level=[]
    for ev in events:
        t0 = ev.time
        first = np.logical_and(train >= t0,
                               train <= t0 + tmax)
        if len(train[first] > 0):
            first_spikes.append(train[first][0] - t0)
            level.append(ev.annotations['Level'])
            
            
    if ax is not None:
        
        ffsp = []
        levell = np.asarray(level)
        A = np.where(levell == 40)
        B = np.where(levell == 50)
        C = np.where(levell == 60)
        D = np.where(levell == 70)
        
        ffsp.append([first_spikes[j] for j in A[0]])
        ffsp.append([first_spikes[j] for j in B[0]])
        ffsp.append([first_spikes[j] for j in C[0]])
        ffsp.append([first_spikes[j] for j in D[0]])
            # first_level = [first_spike(j) for j in np.where(level == lvl)]
        

        ax.boxplot(ffsp, notch=True,
                   flierprops=dict(markerfacecolor='blue', marker='.'))
                # linestyle=" ",
                # marker=".",
                # color='b',
                # markersize=1.)
        
        
def analyze_tone_sequence(segment, unit,
                          trange=None,
                          ax=None,
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
    if trange is None:
        trange = tone_len

    R = np.zeros((len(levels), len(freqs)), dtype=np.float64)
    trial_cnt = np.zeros_like(R)
    for i, ll in enumerate(levels):

        for j, ff in enumerate(freqs):

            for ii, ev in enumerate(events):

                if seq_levels[ii] == ll and seq_freqs[ii] == ff:

                    t0 = ev.time
                    valid = np.logical_and(train >= t0,
                                               train < t0 + trange)
                    R[i, j] += np.sum(valid)
                    trial_cnt[i, j] += 1

    
    R /= (trial_cnt * trange)

#    from scipy.ndimage.filters import gaussian_filter
#    R = gaussian_filter(R, .75)

    if ax is not None:

        ax.imshow(R, vmin=0, vmax=R.max(), aspect='auto',
                  interpolation='nearest', origin='lower')

        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Level (dB SPL)')

        ax.set_xticks(range(0, len(freqs), 4))
        fl = ['%0.1f' % x for x in freqs[::4] / 1000.]
        ax.set_xticklabels(fl)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels(np.asarray(levels, dtype=int))

    return R, freqs, levels

def apply_first_criterion(segs, unit,
                          pre = 0.05,
                          post = 0.05,
                          binwidth = 0.001,
                          **kwargs):

    pre_list = []
    post_list = []
    for seg in segs:
        index = [st.unit for st in seg.spiketrains].index(unit)
        train = seg.spiketrains[index].magnitude
        events = seg.events
        for ev in events:
            t0 = ev.time
            v = np.logical_and(train >= t0 - pre,
                               train < t0)
            w = np.logical_and(train > t0,
                               train <= t0 + post)
            pre_list.append(len(train[v])/pre)
            post_list.append(len(train[w])/post)

    if sum(pre_list) - sum(post_list):
        stat, p = wilcoxon(pre_list, post_list)
    else:
        p = 1
        print('**** P =1 ****')

    return p

def find_response_latency(psth, 
                          pre=50,
                          post=100,
                          binwidth=2,
                          error_range=3,
                          two_bins=False,
                          **kwargs):
    

    n_bins = int(np.ceil((pre + post) / binwidth))
    edges = np.linspace(-pre, post, n_bins+1)
    # edges = -pre + np.arange(n_bins + 1) * binwidth
    # pdb.set_trace()
    spontaneous = psth[:int(pre/binwidth)]
    m = mean(spontaneous)
    std_dev = np.std(spontaneous)
    response_latency = float('NaN')
    
    if two_bins==True:
        twobins = False
        for i, (spikes, times) in enumerate(zip(psth[int(pre/binwidth):
                                                int(pre/binwidth) 
                                                + int(post/binwidth)],
                                                edges[int(pre/binwidth):int(pre
                                                /binwidth) + int(post/binwidth)])):
            if spikes < m-error_range*std_dev or spikes > m+error_range*std_dev:
                if twobins:
                    break
                else:
                    response_latency = times
                    twobins=True
            else:
                twobins=False
    
        if twobins==False:
            response_latency = float('NaN')
            
    else:
        for i, (spikes, times) in enumerate(zip(psth[int(pre/binwidth):
                                                int(pre/binwidth) 
                                                + int(post/binwidth)],
                                                edges[int(pre/binwidth):int(pre
                                                /binwidth) + int(post/binwidth)])):
            if spikes < m-error_range*std_dev or spikes > m+error_range*std_dev:
                response_latency = times
                break

    return response_latency, m, std_dev


def plot_FRA(segment, unit, trange, FRA, ax=None):
    R, freqs, levels = analyze_tone_sequence(segment, unit, trange)
    if FRA is None:
        FRA = R
    else:
        FRA += R
    ax.plot(sum(R)*100/max(sum(R)))
    ax.axis('off')
    ax.set_title('FRA for the first {}ms'.format(
        int(trange*1000)), fontweight="bold")
    return FRA, freqs, levels

def plot_FRA_summary(segment, unit, trange, FRA, ax=None, color = None):
    R, freqs, levels = analyze_tone_sequence(segment, unit, trange)
    if FRA is None:
        FRA = R
    else:
        FRA += R
    ax.plot(sum(R)*100/max(sum(R)), color = color)
    ax.axis('off')
    #ax.set_title('Frequency Tuning', fontweight="bold")
    return FRA, freqs, levels

def compute_signal_power(rec_path, channel_index, unit_index, max_lag):
    from lnpy.metrics import srfpower

    session_path, recording = op.split(rec_path)

    reader = NeoIO(session_path)
    block = reader.read_block(recordings=[recording],
                              dtypes=['spikes'],
                              cluster_groups=['Good', 'MUA'])
    segment = block.segments[0]

    # load stimulus data
    stim_data = helpers.load_drc_data(segment.file_origin,
                                      max_lag=max_lag)

    # get spike train for unit
    unit = [u for u in block.list_units
            if u.annotations['channel_index'] == channel_index and
            u.annotations['unit_index'] == unit_index][0]

    index = [st.unit for st in segment.spiketrains].index(unit)
    spike_train = segment.spiketrains[index].magnitude
    stim_events = segment.events

    # bin spike train (aligned with stimulus)
    rr = helpers.bin_spike_train(spike_train, stim_events,
                                 bin_width=stim_data['dt'],
                                 max_lag=max_lag,
                                 trial_duration=stim_data['trial_duration'])

    signal_power, noise_power, signal_power_err = srfpower(rr['Y'],
                                                           indep_noise=False,
                                                           verbose=False)

    return signal_power, noise_power, signal_power_err

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), frameon=False)

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


def plot_latency_distribution(db_path=None, ax= None, colours = None,
                              latency=20, post=60, binwidth=2,
                              **kwargs):
    all_latencies = np.load(op.join(db_path, 'response/latency_analysis/latencies_4_mice_99.npy'),
                                    encoding="latin1",
                                    allow_pickle=True).tolist()
    n_bins = int(np.ceil((post) / binwidth))
    edges = np.arange(n_bins + 1) * binwidth
    all_latencies_normalised = {}
    for i, mouse in enumerate(sorted(all_latencies)):
        print(mouse)
        latencies = all_latencies[mouse]
        m = 'M'+str(int(i+1)) #mouse[0] + mouse[3:]
        cnt, edges = np.histogram(latencies,
                                  bins=edges)
        percentage = cnt * 100 / len(latencies)
        all_latencies_normalised[m] = list(percentage)
    bar_plot(ax, all_latencies_normalised, colors=colours, total_width=0.8, single_width=1, legend=True)
    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xticklabels([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlim(0, 30)
    ax.set_ylabel('Percentage Cells', fontsize=9)
    ax.set_xlabel('Latency (ms)', fontsize=9)
    ax.axvline(x=latency / 2, color='k', linestyle='--', linewidth=0.7)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    adjust_spines(ax, ['left', 'bottom'])

    return ax

@click.group()
def cli():
    pass


@click.command(name='drc_response')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--period', '-p', default=1.5)
@click.option('--point', '-P', default=45)
def cli_drc_response(db_path=None,
        output=None,
        period=1.5,
        point=45,
        show=False,
        **kwargs):
    '''
    callable function to create a pdf showing all units' responoses to DRC stimuli
    inputs:
    - db_path : database
    - period : the amount of timee in seconds to display on the x-axis
    - point: th etime point around which the plot shouold show the DRC response

    outputs:
    - pdf containing a page pere unit , showing responses to DRC.
    '''
    animals = [op.join(db_path, animal, 'neural') for animal in os.listdir(db_path) if animal.startswith('M102')]
    for animal in animals:
        print(animal)
        file_name = 'DRC_responses_' + animal.split('/')[-2]
        pdf_path = op.join(output, file_name)
        pdf = PdfPages(pdf_path + '.pdf')
        pre_sessions = [op.join(animal, pre_session) for pre_session in os.listdir(animal) if pre_session.startswith('20')]
        for pre_session in sorted(pre_sessions):
            session_path = op.join(pre_session, os.listdir(pre_session)[0])
            print('#######'+session_path)
            recordings = get_drc_segments(session_path)
            print(recordings)
            reader = NeoIO(session_path)
            block = reader.read_block(recordings=recordings,
                                      dtypes=['spikes'],
                                      cluster_groups=['Good', 'MUA'])
            units = block.list_units
            for i, unit in enumerate(units):
                segs = block.segments
                print('analysing: '+unit.file_origin)
                for j, seg in enumerate(segs):
                    fig, axarr = plt.subplots(nrows=3, ncols=1, sharex=True)
                    drc = helpers.load_drc_data(seg.file_origin)['X']
                    drc_new = np.zeros_like(drc)
                    for i, row in enumerate(drc):
                        for j, value in enumerate(row):
                            if value > 0:
                                drc_new[i, j] = 70 + 20 * math.log10(value)
                            else:
                                pass
                    # We want the last second and a half from the trial. there are 2250 bins of 20 ms length each.
                    # (2250x0.020 = 45 sec which is the size of the trial)
                    # 1.5s = 1500ms /20 = 75 bins
                    ax = axarr[0]
                    im1 = ax.imshow(drc_new.T[:, drc_new.shape[0] - 75:],
                                    extent=[43.5, 45., 5, 40],
                                    aspect='auto',
                                    cmap = plt.cm.Greys)
                    ax.set_ylabel('Frequency (kHz)', labelpad=2)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(2, integer=True))
                    fig.colorbar(mappable=im1, ax=axarr, label='Intensity (dB SPL)')
                    ax.set_xlim(43.5,45.0)
                    if seg.spiketrains:
                        index = [st.unit for st in seg.spiketrains].index(unit)
                        train = seg.spiketrains[index].magnitude
                        events = seg.events
                        ax = axarr[1]
                        ax.set_ylabel('Trial', labelpad=2)
                        spike_times_list = []
                        #ax.set_yticks(np.arange(len(events)))
                        #ax.set_yticklabels([str(el) for el in np.arange(1,len(events)+1)])
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
                        for k, ev in enumerate(events):
                            t0 = ev.time
                            v = np.logical_and(train >= t0 + point - period,
                                               train <= t0 + point)
                            spike_times_list.append(train[v] - t0)
                            # instead of appending, plot them
                            spike_times = train[v] - t0

                            ax.vlines(spike_times, k, k + .5, colors=3 * [.1], lw=.5)
                        ax.set_xlim(43.5, 45.0)
                        #ax.set_ylim(0,len(events)+0.5)
                        # make PSTH FOR THE RESPONSE OF THIS UNIT TO THIS DRC TRIAL(S)
                        # on y axis we want the expected amount of spikes per trial and on the x we need 76 bins
                        cnt, edges = np.histogram(np.concatenate(spike_times_list),
                                                  bins=75,
                                                  range=(point - period,
                                                         point))
                        ax = axarr[2]
                        ax.bar(edges[:-1], cnt / len(events), width=edges[1] - edges[0],
                               align='edge', color=3 * [.5])
                        ax.set_xlabel('Time (s)', labelpad=2)
                        ax.set_ylabel('# Spikes/ Trial', labelpad=2)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
                        ax.set_xlim(43.5, 45.0)
                        fig.set_size_inches(7, 5)
                        fig.suptitle(' Chan{: d} Unit{: d} '.format(unit.annotations['channel_index'],
                        unit.annotations['unit_index']) + str(seg.name))

                    #fig.tight_layout(pad=.5, w_pad=1.5)
                    if output is not None:
                        pdf.savefig(fig)

                    plt.close(fig)

        if output is not None:
            pdf.close()

cli.add_command(cli_drc_response)

        
@click.command(name='summary_plot')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--pre', '-p', default=.05)
@click.option('--post', '-P', default=.1)
@click.option('--binwidth', '-b', default=.002)
def cli_summary_plot(db_path=None,
        output=None,
        pre=.05,
        post=.1,
        binwidth=.002,
        alpha = 0.01,
        show=False,
        **kwargs):
    '''
        callable function to create a pdf showing all units' responoses to DRC stimuli
        inputs:
        - db_path : database
        - pre and post: amount of timee peeristimulus to bee shown

        outputs:
        - pdf containing a page pere unit , showing responses to tones and noisebursts as well as statistics about the response.
        '''

    animals = [op.join(db_path, animal) for animal in os.listdir(db_path) if animal.startswith('M102')]
    for animal_path in animals:
        file_name = 'Summary_plot_responses_' + op.split(animal_path)[1]
        pdf_path = op.join(output, file_name)
        pdf = PdfPages(pdf_path + '.pdf')
        sessions = helpers.get_sessions_with_drc_recordings(animal_path)

        for recording in sorted(sessions):
            session_path = sessions[recording]['absolute_path']
            print('analysing session: ', session_path)

            recordings = get_recordings(session_path)
            reader = NeoIO(session_path)

            block = reader.read_block(recordings=recordings,
                                      dtypes=['spikes'],
                                      cluster_groups=['Good', 'MUA'])

            units = block.list_units
            n_units = len(units)

            if n_units > 0:
                for i in units:

                    print('Session {} Chan{:d} Unit{:d}'.format(
                        session_path.split('/')[-2],
                        i.annotations['channel_index'],
                        i.annotations['unit_index']))

                    grid = plt.GridSpec(5, 18)
                    axes = np.array([plt.subplot(grid[:5, :4]),
                                     plt.subplot(grid[:5, 5:13]),
                                     plt.subplot(grid[0:2, 14:]),
                                     plt.subplot(grid[2:5, 14:])])
                    fig = plt.gcf()

                    psth_tone = np.zeros(int(np.ceil((pre + post) / binwidth)))

                    n_events_tone = 0

                    # FRA_early = None
                    FRA_late = None
                    tone_segs = []
                    for seg in block.segments:
                        if 'ToneSequence' in seg.name:
                            ps, edges, n_events, duration_tone = compute_psth(seg,
                                         i,
                                         binwidth=binwidth,
                                         pre=pre,
                                         post=post,
                                         ax=None)
                            tone_segs.append(seg)

                            psth_tone += ps
                            n_events_tone += n_events

                            FRA_late, freqs, levels = plot_FRA_summary(seg, i,
                                                0.1,
                                                FRA_late,
                                                ax=axes[2])

                    if len(tone_segs) > 0:
                        p = apply_first_criterion(tone_segs, i)

                    # plot PSTH Tone
                    ax = axes[1]
                    ax.set_title('Tone PSTH', fontweight="bold")
                    ax.set_xlabel('Spike rate')
                    ax.set_ylabel('Spike rate')
                    ax.bar(edges[:-1], psth_tone / binwidth / n_events_tone,
                           width=binwidth,
                           color=3 * [.5],
                           edgecolor='none',
                           linewidth=0,
                           align='edge')

                    if p <= alpha:
                        colour = scu.NICE_COLORS['forest green']
                    else:
                        colour = scu.NICE_COLORS['tomato']

                    ax.axvspan(0, duration_tone,
                               color=colour,
                               alpha=.5,
                               lw=0,
                               zorder=0)
                    ax.text(post, (0.9 * psth_tone/ binwidth / n_events_tone).max(),
                                   "resp. p= " + '%s' % float('%.3g' % p), horizontalalignment='right')

                    resp, m, std = find_response_latency(
                        psth_tone / binwidth / n_events_tone)

                    if not math.isnan(resp):
                        ax.axvline(x=resp, color='r', linewidth = 1.0)
                        resp = float(resp) * 1000
                        ax.text(post, (0.8 * psth_tone/ binwidth
                                        / n_events_tone).max(), "latency= " + '%s'
                                        % float('%.3g' % resp) + " ms",
                                        horizontalalignment='right')

                    ax.axhline(m, color='k', linestyle='--')
                    ax.axhspan(m-3*std, m+3*std,
                               color=scu.NICE_COLORS['lightgray'],
                               alpha=.5,
                               lw=0,
                               zorder=0)

                    # plot waveform
                    ax = axes[0]
                    plot_waveforms(ax, i)

                    # plot FRA_late
                    if FRA_late is not None:
                        ax=axes[3]
                        ax.imshow(FRA_late, vmin=0, vmax=FRA_late.max(), aspect='auto',
                          interpolation='nearest', origin='lower')

                        ax.set_xlabel('Frequency (kHz)')
                        ax.set_ylabel('Level (dB)')

                        ax.set_xticks(range(0, len(freqs), 4))
                        fl = ['%0.1f' % x for x in freqs[::4] / 1000.]
                        ax.set_xticklabels(fl)
                        ax.set_yticks(range(len(levels)))
                        ax.set_yticklabels(np.asarray(levels, dtype=int))


                    fig.suptitle('Session {} Chan{:d} Unit{:d}'.format(
                        session_path.split('/')[-2],
                        i.annotations['channel_index'],
                        i.annotations['unit_index']),
                        color=get_clustergroup_color(i.description), x=0.1, y=.95,
                        horizontalalignment='left', verticalalignment='top')

                    for axx in axes.flat:
                        if axx is not None:
                            scu.set_font_axes(axx, add_size=2)
                            scu.simple_xy_axes(axx)

                    fig.set_size_inches(17*(2/3),4*(2/3))  # 1.5 * n_rows)
                    rect = (0.05, 0.025, 0.95, 0.95)
                    fig.tight_layout(pad=.1, h_pad=.075, w_pad=.025, rect=rect)


                    if output is not None:
                        pdf.savefig(fig)

                    plt.close(fig)

        if output is not None:
            pdf.close()
        
cli.add_command(cli_summary_plot)


@click.command(name='comparison')
@click.argument('animal_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--pre', '-p', default=.05)
@click.option('--post', '-P', default=.1)
@click.option('--binwidth', '-b', default=.002)
@click.option('--error_range', '-e', default=3)
@click.option('--two_bins', '-t', is_flag=True)
def cli_comparison(animal_path=None,
        output=None,
        pre=.05,
        post=.1,
        binwidth=.002,
        alpha = 0.01,
        show=False,
        error_range = 3,
        two_bins = True,
        **kwargs): 
    
    if two_bins:
        tbcriteria = 'two bins'
    else:
        tbcriteria = 'one bin'
        
    file_name = 'comparison_of_criteria_' + str(binwidth) +'_'+ str(error_range) + '_' + tbcriteria + '_' + animal_path.split('/')[-1]
    
    units_dict = {}
    
    sessions = [session for session in os.listdir(animal_path) if not session.startswith('_')]
    
    total_units = 0
    n_core_units = 0
    non_core = 0
    nans = 0
    for rec_session in sorted(sessions):
        session_path = op.join(animal_path, rec_session, os.listdir(op.join(animal_path,
                                                            rec_session))[0])
        recordings = get_recordings(session_path)
        reader = NeoIO(session_path)
        
        block = reader.read_block(recordings=recordings,
                                  dtypes=['spikes'],
                                  cluster_groups=['Good', 'MUA'])
            
        units = block.list_units
        n_units = len(units)
        total_units += n_units
        # units_dict['n_units'] = n_units
        
        if n_units > 0:
            for i in units:
                
                print('Session {} Chan{:d} Unit{:d}'.format(
                    session_path.split('/')[-2], 
                    i.annotations['channel_index'],
                    i.annotations['unit_index']))
              
                psth_tone = np.zeros(int(np.ceil((pre + post) / binwidth)))

                n_events_tone = 0
                
                # FRA_early = None
                FRA_late = None
                tone_segs = []
                for seg in block.segments:
                    if 'ToneSequence' in seg.name:
                        ps, edges, n_events, duration_tone = compute_psth(seg,
                                     i,
                                     binwidth=binwidth,
                                     pre=pre,
                                     post=post,
                                     ax=None)
                        tone_segs.append(seg)
                        
                        psth_tone += ps
                        n_events_tone += n_events
                        
                        
                if len(tone_segs) > 0:
                    p = apply_first_criterion(tone_segs, i)
                     
                resp, m, std = find_response_latency(
                    psth_tone / binwidth / n_events_tone,
                    binwidth = binwidth,
                    error_range = error_range,
                    two_bins = two_bins)
                
                if p <= alpha: #and resp <= 20:
                    if resp <= 0.02:
                        n_core_units += 1
                    elif resp > 0.02:
                        non_core += 1
                    elif math.isnan(resp):
                        nans += 1
                
    units_dict['n_units'] = total_units
    units_dict['n_core']= n_core_units
    units_dict['non_core']= non_core
    units_dict['nans']= nans
    
    fig, ax = plt.subplots()
    ax.bar(units_dict.keys(), units_dict.values())
    ax.set_title('Core Units '+ str(binwidth) +' '+ str(error_range) + ' ' + tbcriteria)
    

    for p in ax.patches:
             ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                 textcoords='offset points')
    
    
    fig.savefig(output + file_name,
            format='pdf',
            dpi=300)
    
cli.add_command(cli_comparison)


# -----------------------------------------------------------------------------------------
# make graph which summarises the number of cells detected across session, per tetrode
# -----------------------------------------------------------------------------------------
@click.command(name='tetrode_summary_plot')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--pre', '-p', default=.05)
@click.option('--post', '-P', default=.1)
@click.option('--binwidth', '-b', default=.002)
def cli_tetrode_summary_plot(db_path=None,
        output=None,
        pre=.05,
        post=.1,
        binwidth=.002,
        alpha = 0.01,
        show=False,
        **kwargs):
    '''
        callable function to create a pdf showing all units' responoses to DRC stimuli
        inputs:
        - db_path : database
        - pre, post, binwidth: params to construct histograms of unit responses.

        outputs:
        - pdf containing a page pere unit , showing responses to all stimuli + summary statistics whcih recognise CORE units
        '''

    animals = [op.join(db_path, animal) for animal in os.listdir(db_path) if animal.startswith('M102')]
    for animal in animals:

        core_units = np.load(op.join('/home/marios/Sdrive/DBIO_LindenLab_DATA/DATA_ANAT/jules/analysis/response',
                                     op.split(animal)[1] , op.split(animal)[1])+'_units_core.npy',
                             allow_pickle=True, encoding="latin1")
        days_recs = sorted(os.listdir(op.join(animal,'neural')))
        pre_session_paths = [op.join(animal,'neural', day) for day in days_recs if day.startswith('20')]
        session_paths = [op.join(pre_session_path, os.listdir(pre_session_path)[0]) for pre_session_path in pre_session_paths]
        units_total_core = np.zeros((18, len(session_paths)))

        for j, session_path in enumerate(session_paths):
            units_total_core[0,j] = op.split(op.split(session_path)[0])[1]
            print('analyzing:', session_path)
            recordings = helpers.get_recordings(session_path)
            print(recordings)
            reader = NeoIO(session_path)
            block = reader.read_block(recordings=recordings,
                                      dtypes=['spikes'],
                                      cluster_groups=['Good', 'MUA'])
            #pdb.set_trace()

            adv = np.load(op.join(session_path, 'advancer_data.npz'), allow_pickle=True, encoding="latin1")
            units_total_core[1,j] = adv['channelgroup_1'].tolist()['Depth']

            for i, tetrode in enumerate(block.channel_indexes):
                n_units = len(tetrode.units)
                units_total_core[i+2,j]=n_units

        #make nested list with session name depth and number of units for each date present in the core_units
        list_of_sessions = []
        for item in core_units:
            unit = item.as_dict()
            list_of_sessions.append(unit['session'] +'_'+str(unit['tetrode']))

        session_dict = Counter(list_of_sessions)
        for i, sesh in enumerate(units_total_core[0,:]):
            for j, rec in enumerate(session_dict.keys()):
                if float(rec[:10]) == sesh:
                    units_total_core[int(rec[-1])+9,i] = session_dict[rec]

        #plot it all
        file_name = 'tetrode_summary_' + op.split(animal)[1]
        pdf_path = op.join(output, op.split(animal)[1], file_name)
        pdf = PdfPages(pdf_path + '.pdf')
        #pdb.set_trace()
        for i in range(8):
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(range(units_total_core.shape[1]), units_total_core[i+2,:], 'ko', label = 'Total')
            ax.plot(range(units_total_core.shape[1]), units_total_core[i+10,:], 'rx', label = 'Core')
            ax.set_xticks(range(units_total_core.shape[1]))
            ax.set_xticklabels([str(l)[:-2] for l in units_total_core[0,:]], rotation = 90)
            ax.set_xlabel('Recording')
            ax.set_ylabel('N of units')
            ax2 = ax.twinx()
            ax2.plot(range(units_total_core.shape[1]), units_total_core[1,:], 'b-', label='Depth')
            ax2.set_ylabel('Depth of electrodes (um)')
            ax.legend()
            fig.tight_layout()
            if output is not None:
                pdf.savefig(fig)

            plt.close(fig)

        if output is not None:
            pdf.close()

cli.add_command(cli_tetrode_summary_plot)


######----------------------------------------------------
# Function to plot the efiltering effect of each criterion and the overlap between them
# could also add the bar plot showing the total/ core cell count on another subplot?
######----------------------------------------------------
@click.command(name='venn')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--latency', '-l', default = 20)
@click.option('--alpha', '-a', default = 0.01)
@click.option('--pre', '-p', default=50)
@click.option('--post', '-P', default=100)
@click.option('--binwidth', '-b', default=2)
def cli_venn(db_path=None,
             output=None,
             latency = 20,
             alpha = 0.01,
             pre = 50,
             post = 100,
             binwidth = 2,
             **kwargs):
    '''
        callable function to create a pdf showing all units' responoses to DRC stimuli
        inputs:
        - db_path : database
        - period : the amount of timee in seconds to display on the x-axis
        - point: th etime point around which the plot shouold show the DRC response

        outputs:
        - pdf containing a page pere unit , showing responses to DRC.
    '''
    animals = [op.join(db_path, animal) for animal in os.listdir(db_path) if animal.startswith('M102')]
    #create pdf doc
    file_name = 'Venn_diagrams_for_3_criteria_' + str(latency) +'_'+str(alpha)
    pdf_path = op.join(output, file_name)
    pdf = PdfPages(pdf_path + '.pdf')
    all_latencies

    for animal in sorted(animals):
        latencies = []
        # days_recs = sorted(os.listdir(op.join(animal, 'neural')))
        # pre_session_paths = [op.join(animal, 'neural', day) for day in days_recs if day.startswith('20')]
        # session_paths = [op.join(pre_session_path, os.listdir(pre_session_path)[0]) for pre_session_path in
        #                  pre_session_paths]
        sessions = helpers.get_sessions_with_drc_recordings(animal)
        Abc = 0
        ABc = 0
        AbC = 0
        aBc = 0
        aBC = 0
        abC = 0
        ABC = 0
        abc = 0

        # load matches and get neurons which are duplicates
        matched = np.load(op.join('/home/marios/Sdrive/DBIO_LindenLab_DATA/DATA_ANAT/jules/analysis/EM_waveform_matching/250_microns/99_CI',
                    op.split(animal)[1], 'matched_units_waveforms.npz'),
            encoding="latin1", allow_pickle=True)
        matched_indices = matched['thresholded_indices']
        matched_units = []
        if matched_indices.shape[0] > 0:
            matched_units.append(list(matched_indices[0]))

        if matched_indices.shape[0] > 0:
            for i in range(1, len(matched_indices)):
                for n in range(len(matched_units)):
                    if matched_indices[i][0] in matched_units[n]:
                        matched_units[n].append(matched_indices[i][1])
                        break
                    elif matched_indices[i][1] in matched_units[n]:
                        matched_units[n].append(matched_indices[i][0])
                        break
                    elif n == len(matched_units) - 1:
                        matched_units.append(list(matched_indices[i]))

            matched_neurons = []
            for group in matched_units:
                no_duplicates_group = sorted(list(set(group)))
                matched_neurons.append(no_duplicates_group)

            neuron_duplicate = []
            for group in matched_neurons:
                neuron_duplicate.append(group[1:])
        else:
            matched_neurons = []
            neuron_duplicate = []
        duplicates = []
        for kk in neuron_duplicate:
            for nn in kk:
                duplicates.append(int(nn))
        absolute_index = -1

        for j, session in enumerate(sorted(sessions)):
            session_path = sessions[session]['absolute_path']
            print('analysing session: ', session_path)
            recordings = helpers.get_recordings(session_path)
            reader = NeoIO(session_path)
            block = reader.read_block(recordings=recordings,
                                      dtypes=['spikes'],
                                      cluster_groups=['Good', 'MUA'])

            units = block.list_units

            if len(units) > 0:
                for i in units:
                    absolute_index += 1
                    print('Session {} Chan{:d} Unit{:d}'.format(
                        session_path.split('/')[-2],
                        i.annotations['channel_index'],
                        i.annotations['unit_index']))

                    psth_tone = np.zeros(int(np.ceil((pre + post) / binwidth)))

                    n_events_tone = 0

                    tone_segs = []
                    for seg in block.segments:
                        if 'ToneSequence' in seg.name:
                            ps, edges, n_events, duration_tone = compute_psth(seg,
                                                                              i,
                                                                              binwidth=binwidth,
                                                                              pre=pre,
                                                                              post=post,
                                                                              ax=None)
                            tone_segs.append(seg)

                            psth_tone += ps
                            n_events_tone += n_events

                    if len(tone_segs) > 0:
                        p = apply_first_criterion(tone_segs, i)
                        A = p <= alpha
                    resp, m, std = find_response_latency(
                        psth_tone / binwidth / n_events_tone)
                    B = resp <= latency

                    if A:
                        if absolute_index not in duplicates:
                            latencies.append(resp)
                            print(latencies)
                        else:
                            pass
                    else:
                        pass

                    #now find signal power
                    context_path = op.join('/home/marios/Sdrive/DBIO_LindenLab_DATA/DATA_ANAT/jules/analysis/context/ALS_MATLAB/context_30',
                                           op.split(animal)[1])
                    model_file = '{}_{}_{}_chan{:02d}_unit{:02d}.npy'.format(
                        block.file_origin.split('/')[-4], block.file_origin.split('/')[-2],\
                        block.file_origin.split('/')[-1], i.annotations['channel_index'],\
                        i.annotations['unit_index'])
                    if op.isfile(op.join(context_path,model_file)):
                        context_data = np.load(op.join(context_path, model_file),
                                               allow_pickle=True,
                                               encoding="latin1").tolist()
                        C = context_data['stats']['signal_power'] - context_data['stats']['error_signal'] > 0

                        if A and not B and not C:
                            Abc += 1
                        elif A and B and not C:
                            ABc += 1
                        elif A and not B and C:
                            AbC += 1
                        elif A and B and C:
                            ABC += 1
                        elif not A and B and not C:
                            aBc += 1
                        elif not A and B and C:
                            aBC += 1
                        elif not A and not B and C:
                            abC += 1
                        elif not A and not B and not C:
                            abc += 1
        all_latencies[op.split(animal)[1]] = latencies
        #pdb.set_trace()
        #plot it all into a venn3 diagram for this mouse
        fig, axar = plt.subplots(nrows = 1, ncols=2)
        fig.suptitle(str(op.split(animal)[1]) + ' core cell criteria breakdown')
        ax = axar[1]
        ax = venn3(subsets=(Abc, aBc, ABc, abC, AbC, aBC, ABC), set_labels=('Tone-responsiveness', 'Latency <='+ str(latency),
                                                                            'Signal Power'))

        ax = axar[0]
        x= range(2)
        y = [Abc + aBc + ABc + abC + AbC + aBC + ABC + abc, ABC]
        ax.bar(x,y, alpha = 0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['Total', 'Core'])
        for index, value in enumerate(y):
            ax.text(index, value, str(value))

        fig.set_size_inches(15,10)
        fig.tight_layout()
        pdf.savefig(fig)
    pdf.close()
    np.save(op.join(output, 'latencies_4_mice_new_99'), all_latencies)

cli.add_command(cli_venn)

# function to give us the numbers of core/ core+nearby and total number of cell count for each animal

@click.command(name='response_new')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--pre', '-p', default=50)
@click.option('--post', '-P', default=100)
@click.option('--binwidth', '-b', default=2)
def cli_response_new(db_path=None,
        output=None,
        pre=50,
        post=100,
        binwidth=2,
        alpha = 0.01,
        latency = 20,
        **kwargs):
    '''
        callable function to get a newe updated database only with the Core units which have short latencies and also aree responsive to tones.
        inputs:
        - db_path : database
        - pre, post, binwidth: params for the analysis
        - alpha: confidence level for responsiveness of units to tones
        - latency: minimum latency for inclusion

        outputs:
        - new database in the form of an .npy file
        '''

    animals = [op.join(db_path, animal) for animal in os.listdir(db_path)
               if animal.startswith('M102') and not animal.endswith('_neural')]
    all_latencies = {}
    for animal in sorted(animals):
        latencies = []
        data = []
        index = 0
        absolute_index = 0
        core_nearby = 0
        core = 0
        total = 0
        sessions = helpers.get_sessions_with_drc_recordings(animal)

        all_cells = []
        # load matches and get neurons which are duplicates
        matched = op.join('/home/marios/data/bbsrc/analysis/waveform_matching/99%CI',
                        op.split(animal)[1], 'matched_units_waveforms.npz')
        if op.exists(matched):
            matched = np.load(matched, encoding="latin1", allow_pickle=True)
            matched_indices = matched['thresholded_indices']
            matched_units = []
            if matched_indices.shape[0] > 0:
                matched_units.append(list(matched_indices[0]))

            if matched_indices.shape[0] > 0:
                for i in range(1, len(matched_indices)):
                    for n in range(len(matched_units)):
                        if matched_indices[i][0] in matched_units[n]:
                            matched_units[n].append(matched_indices[i][1])
                            break
                        elif matched_indices[i][1] in matched_units[n]:
                            matched_units[n].append(matched_indices[i][0])
                            break
                        elif n == len(matched_units) - 1:
                            matched_units.append(list(matched_indices[i]))

                matched_neurons = []
                for group in matched_units:
                    no_duplicates_group = sorted(list(set(group)))
                    matched_neurons.append(no_duplicates_group)

                neuron_duplicate = []
                for group in matched_neurons:
                    neuron_duplicate.append(group[1:])
            else:
                matched_neurons = []
                neuron_duplicate = []

            duplicates = []
            for i in neuron_duplicate:
                for j in i:
                    duplicates.append(int(j))
        else:
            duplicates = []

        for recording in sorted(sessions):
            session = sessions[recording]['absolute_path']
            print('analysing session: ', session)
            recordings = helpers.get_recordings(session)
            reader = NeoIO(session)
            block = reader.read_block(recordings=recordings,
                                      dtypes=['spikes'],
                                      cluster_groups=['Good', 'MUA'])
            adv = np.load(op.join(session, 'advancer_data.npz'), allow_pickle=True, encoding="latin1")
            depth = adv['channelgroup_1'].tolist()['Depth']
            # loop overe all the tetrodes and create a list for each one containing tuples with absolute index and the relevant cell obj
            for i, tetrode in enumerate(block.channel_indexes):
                tet_list = []
                for k, cell in enumerate(tetrode.units):
                    cell.annotations['session'] = session.split('/')[-2]
                    tet_list.append((absolute_index, cell))
                    absolute_index += 1
                all_cells.append(tet_list)
        for tt, tet in enumerate(all_cells):
            if tet:
                for group in tet:
                    neuron = group[1]
                    print('analysing: ', neuron.file_origin)
                    absolute_index = group[0]
                    psth_tone = np.zeros(int(np.ceil((pre + post) / binwidth)))
                    n_events_tone = 0
                    segs = [st.segment for st in neuron.spiketrains]
                    #make a list comprehension which takes the first two tone seq segs from a sorted list
                    tone_segs = [tone_seg for tone_seg in segs if 'ToneSequence' in tone_seg.name]
                    if len(tone_segs)>2:
                        tone_segs = tone_segs[:2]
                        print([seg.name for seg in tone_segs])
                    for seg in tone_segs:
                        try:
                            ps, edges, n_events, duration_tone = compute_psth(seg,
                                                                              neuron,
                                                                              binwidth=binwidth,
                                                                              pre=pre,
                                                                              post=post,
                                                                              ax=None)
                            psth_tone += ps
                            n_events_tone += n_events
                        except:
                            print('THIS SEGMENT MAY BE EMPTY ')
                    if len(tone_segs) > 0:
                        p = apply_first_criterion(tone_segs, neuron)

                    resp, m, std = find_response_latency(
                        psth_tone / binwidth / n_events_tone)
                    #if the criteria are met then loop over the wholee tetrode and append everything
                    # this is to include the nearby cells
                    if p <= alpha and resp <= latency:
                        for neur in tet:
                            unit = neur[1]
                            absolute_index = neur[0]
                            data.append(helpers.SpikeUnit(tetrode=unit.annotations['channel_index'],
                                                          cluster=unit.annotations['unit_index'],
                                                          waveforms=unit.annotations['waveform']['mean'],
                                                          cluster_group=unit.annotations['group'],
                                                          index=index,
                                                          absolute_index=absolute_index,
                                                          depth=depth,
                                                          session=unit.file_origin,
                                                          core = true ))
                            index += 1
                            if absolute_index not in duplicates:
                                core_nearby += 1
                            else:
                                pass
                        break
                    else:
                        data.append(helpers.SpikeUnit(tetrode=unit.annotations['channel_index'],
                                                      cluster=unit.annotations['unit_index'],
                                                      waveforms=unit.annotations['waveform']['mean'],
                                                      cluster_group=unit.annotations['group'],
                                                      index=index,
                                                      absolute_index=absolute_index,
                                                      depth=depth,
                                                      session=unit.file_origin,
                                                      core=false))

        data.append({'core_nearby' : core_nearby})
# now loop again to get the total number of ells after matches subtracted and also the ONLY CORE  cells
        for tet in all_cells:
            if tet:
                for group in tet:
                    total += 1
                    neuron = group[1]
                    print('analysing: ', neuron.file_origin)
                    absolute_index = group[0]
                    psth_tone = np.zeros(int(np.ceil((pre + post) / binwidth)))
                    n_events_tone = 0
                    segs = [st.segment for st in neuron.spiketrains]
                    tone_segs = [tone_seg for tone_seg in segs if 'ToneSequence' in tone_seg.name]
                    if len(tone_segs) > 2:
                        tone_segs = tone_segs[:2]
                        print([seg.name for seg in tone_segs])
                    for seg in tone_segs:
                        try:
                            ps, edges, n_events, duration_tone = compute_psth(seg,
                                                                              neuron,
                                                                              binwidth=binwidth,
                                                                              pre=pre,
                                                                              post=post,
                                                                              ax=None)
                            psth_tone += ps
                            n_events_tone += n_events
                        except:
                            print('THIS SEGMENT MAY BE EMPTY ')
                    if len(tone_segs) > 0:
                        p = apply_first_criterion(tone_segs, neuron)

                    resp, m, std = find_response_latency(
                        psth_tone / binwidth / n_events_tone)
                    if p <= alpha and absolute_index not in duplicates:
                        latencies.append(resp)
                        if resp <= 20.:
                            core += 1
                            index += 1

        all_latencies[op.split(animal)[1]] = latencies #this will be used by the figure function to plot panel D

        data.append({'total' : total - len(duplicates)})
        data.append({'core' : core })

        np.save(op.join(output, op.split(animal)[1], op.split(animal)[1] + '_units_core_nearby_99'), data)
    np.save(op.join(output, 'latency_analysis', 'latencies_4_mice_99'), all_latencies)

cli.add_command(cli_response_new)

