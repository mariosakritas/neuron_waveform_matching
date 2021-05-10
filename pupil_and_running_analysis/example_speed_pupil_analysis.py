#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyse speed and pupil in example sessions
@author: marios
"""


from __future__ import print_function

import os
import os.path as op
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from datetime import datetime
from scipy import signal
from scipy import interpolate
from scipy import stats
import traceback
from stimcontrol.io.database import NeoIO
from stimcontrol.util import makedirs_save
import stimcontrol.util as scu
import statsmodels.api as sm
import helpers
import pdb


def filter_data(x, fs, f_cutoff=5., order=2):

    Wn = f_cutoff / fs * 2
    b, a = signal.butter(order, Wn,
                         btype='lowpass',
                         analog=False,
                         output='ba')

    data = signal.filtfilt(b, a, x, axis=0)

    return data

def load_cylinder_running_data(seg_path,
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
    data = np.load(op.join(seg_path, 'ttl_events.npz'), encoding="latin1")

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

def load_pupil_data(seg_path, smooth_len=.3):
    data = np.load(op.join(seg_path,'rpi_camera_2_pupil_data.npy'),
                   encoding="latin1", allow_pickle=True).item()

    video_ts = data['timestamps']  # video time stamps aligned to the neural data (in seconds)
    median_eye_length = np.nanmedian(data['eye_length'])
    pupil = data['pupil_size_pix']
    # REMOVE THIS DIVISION BY '2' HERE WHEN YOU PRODUCE NEW PUPIL FILES WITH THE RIGHT AXES DATA
    pupil = np.divide(pupil, 2*median_eye_length)
    # the length of the larger ellipse axis in pixels normalised by the length of the eye
    # smooth pupil trace (needs more work)
    dt = np.median(np.diff(video_ts))
    n_win = max(int(round(smooth_len / dt)), 1)
    if n_win % 2 == 0:
        # make sure window is symmetric
        n_win += 1
    win = signal.gaussian(n_win, 1.5)

    pupil = np.convolve(pupil, win / win.sum(), 'same')

    return video_ts, pupil

@click.group()
def cli():
    pass

@click.command(name='segment')
@click.argument('seg_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--show', '-s', is_flag=True)
def cli_segment(seg_path=None,
                    output=None,
                    show=False,
                    **kwargs):

    t_speed, backward, forward, speed = load_cylinder_running_data(seg_path)
    t_video, pupil = load_pupil_data(seg_path)

    fig, axarr = plt.subplots(nrows=2, ncols=2, sharex='col')

    ax = axarr[0,0]
    ax.plot(t_speed, speed)
    ax.set_title('Velocity trace: ' + op.split(seg_path)[1].split("_")[0] + ' ' + op.split(seg_path)[1].split("_")[1]
                 +' ' + op.split(seg_path)[1].split("_")[2], fontsize =10)
    ax.set_xlabel('Time (s)', fontsize= 8)
    ax.set_ylabel('Running speed (cm/s)', fontsize= 8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.set_ylim(0, max(speed)+ 5/100 *max(speed))

    ax = axarr[1,0]
    ax.plot(t_video, pupil)
    ax.set_title('Pupil trace: ' + op.split(seg_path)[1].split("_")[0] + ' ' + op.split(seg_path)[1].split("_")[1]
                 + ' ' + op.split(seg_path)[1].split("_")[2], fontsize= 10)
    ax.set_xlabel('Time (s)', fontsize= 8)
    ax.set_ylabel('Pupil size (pixels)', fontsize= 8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.set_ylim(0, max(pupil)+ 5/100 *max(pupil))

    #PLOT SCATTER PLOT FOR SPEED AGAINST PUPIL
    #interpolate to get same number of data points
    f1 = interpolate.interp1d(t_video, pupil,
                              kind='linear',
                              bounds_error=False,
                              fill_value=np.NaN)

    f2 = interpolate.interp1d(t_speed, speed,
                              kind='linear',
                              bounds_error=False,
                              fill_value=0)

    # create new time frame
    newt = np.arange(0.10, 675., 0.20)
    new_pupil = f1(newt)
    new_speed = f2(newt)

    ax = axarr[0,1]
    ax.scatter(new_speed, new_pupil, s=0.7)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(150, 300)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Running speed (cm/s)', fontsize=8)
    ax.set_ylabel('Pupil size (pixels)', fontsize=8)

    #PLOT HISTOGRAM FOR PUPIL DURING RUNNING AND NON-RUNNING. (CHOOSE THRESHOLD)
    # running_threshold = 1.0
    #
    # speed_pupil = np.zeros((len(new_pupil),2))
    # for i, (spd, pup) in enumerate(zip(new_speed, new_pupil)):
    #     if spd >= 1.0:
    #         speed_pupil[i,1]=pup
    #         s

        # for j, t0 in enumerate(stim_time[v]):  # looping over trials
        #
        #     X[j, :] = f1(t0 + t_grid)
        #     S[j, :] = f2(t0 + t_grid)
        #
        #     # mean pupil size of the pre-event time.
        #     norm_trials[j] = np.mean(X[j, :n_pre])
        #
        #
        # if normalize:
        #     # this approach is using broadcasting; for details see
        #     # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
        #     X = (X.T / norm_trials).T
        #
        # norm_cond_trials[cond] = norm_trials
        #
        # is_running = np.max(S, axis=1) > running_threshold
        # run_trials = X[is_running, :]
        # norun_trials = X[~is_running, :]
        #
        # print('run=', run_trials.shape[0])
        # print('still=', norun_trials.shape[0])
    if output == None:

        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        for ff in ['png', 'pdf']:
            fig.savefig(op.join(output, 'speed_pupil_' + op.split(seg_path)[1]+'.'+ ff),
                        format=ff,
                        dpi=300)


cli.add_command(cli_segment)


@click.command(name='pupil-running-pdfs')
@click.argument('database', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--show', '-s', is_flag=True)
def cli_pupil_running_pdfs(database=None,
                    output=None,
                    show=False,
                    **kwargs):
    #code to create a pdf per animal containing the pupil and running summaries for each DRC segment in the session
    animals = [op.join(database, animal) for animal in os.listdir(database) if animal.startswith('M102') and not animal.endswith('_neural')]
    for animal in sorted(animals):
        #create pdf
        if output is not None:
            # Create pdf files to save the plots
            new_output = op.join(output, op.split(animal)[1])
            if not op.exists(new_output):
                os.makedirs(new_output)
            file_name = op.split(animal)[1] + '_pupil_running_traces'
            pdf_path = op.join(new_output, file_name)
            pdf = PdfPages(pdf_path + '.pdf')
        #find sessions with drc segments
        sessions = helpers.get_sessions_with_drc_recordings(animal)
        for session in sorted(sessions):
            session = sessions[session]
            #create a figure to take the 3 subplots
            fig, axar = plt.subplots(nrows = 2*len(session['recordings']) , ncols = 2)
            # find the drc segments
            if len(session['recordings']) > 0:
                for i, recording in enumerate(sorted(session['recordings'])):
                    rec = op.join(session['absolute_path'], recording)
                    print(rec)
                    try:
                        # plot running
                        t, forw, back, speed = load_cylinder_running_data(rec)
                        video_ts, pupil = load_pupil_data(rec)

                        ax = axar[0 + i * 2, 0]
                        ax.plot(t, speed)
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Speed (cm/s)')
                        ax.xaxis.set_major_locator(MaxNLocator(5))
                        ax.yaxis.set_major_locator(MaxNLocator(4))

                        ax = axar[1 + i * 2, 0]
                        ax.plot(video_ts, pupil)
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Pupil (rel. eye width)')
                        ax.xaxis.set_major_locator(MaxNLocator(5))
                        ax.yaxis.set_major_locator(MaxNLocator(4))

                        ax = axar[1 + i * 2, 1]
                        # remove nans
                        hist_pupil = pupil[~np.isnan(pupil)]
                        ax.hist(hist_pupil)
                        ax.set_xlabel('Pupil (rel. eye width)')
                        ax.set_ylabel('# time-points')
                        ax.xaxis.set_major_locator(MaxNLocator(5))
                        ax.yaxis.set_major_locator(MaxNLocator(4))

                        # scatter the pupil and the running data
                        # interpolate to get same number of data points
                        f1 = interpolate.interp1d(video_ts, pupil,
                                                  kind='linear',
                                                  bounds_error=False,
                                                  fill_value=0.0)

                        f2 = interpolate.interp1d(t, speed,
                                                  kind='linear',
                                                  bounds_error=False,
                                                  fill_value=np.NaN)

                        # create new time frame
                        newt = np.arange(0.10, 675., 0.20)
                        new_pupil = f1(newt)
                        new_speed = f2(newt)

                        run = new_speed >= 1.0
                        for m, (t, sp) in enumerate(zip(newt, new_speed)):
                            if sp < 1.0:
                                pass
                            elif sp >= 1.0:
                                # get the indeces of the elements between t =t and t=t+10s
                                indeces = np.where((newt >= t) & (newt < t + 10.0))
                                run[indeces] = True
                        run = ~np.isnan(new_pupil) & ~np.isnan(new_speed) & run
                        norun = ~np.isnan(new_pupil) & ~np.isnan(new_speed) & ~run

                        ax = axar[0 + i * 2, 1]
                        ax.scatter(new_speed[norun], new_pupil[norun], s = 0.6, alpha = 0.5)
                        ax.scatter(new_speed[run], new_pupil[run], c = 'g', s = 0.6, alpha = 0.5)

                        ax.set_xlabel('Running speed (cm/s)', fontsize=8)
                        ax.set_ylabel('Pupil size (pixels)', fontsize=8)
                        ax.xaxis.set_major_locator(MaxNLocator(5))
                        ax.yaxis.set_major_locator(MaxNLocator(4))

                    except Exception:
                        traceback.print_exc()
                        print('Something missing in this segment')
            for ax in axar.flat:
                right_side = ax.spines["right"]
                right_side.set_visible(False)
                top_side = ax.spines["top"]
                top_side.set_visible(False)
                helpers.adjust_spines(ax, ['bottom', 'left'])
            fig.suptitle(recording)
            fig.set_size_inches(7, 9)
            plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.85, wspace=0.4, hspace=0.15)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()

cli.add_command(cli_pupil_running_pdfs)

@click.command(name='overall-pupil')
@click.argument('database', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--show', '-s', is_flag=True)
def cli_overall_pupil(database=None,
                    output=None,
                    show=False,
                    **kwargs):
    #code to create a distribution of pupil sizes for each animal and for all animals; plotted on the same figure
    animals = [op.join(database, animal, 'neural') for animal in os.listdir(database) if animal.startswith('M102') and not animal.endswith('_neural')]
    all_pupil = []
    fig, axar = plt.subplots(nrows=5, ncols=1, sharex=True)
    for i, animal in enumerate(sorted(animals)):
        animal_pupil = []
        #find sessions with drc segments
        sessions = helpers.get_sessions_with_drc_recordings(animal)
        for session in sorted(sessions):
            session = sessions[session]
            # find the drc segments
            if len(session['recordings']) > 0:
                for j, recording in enumerate(sorted(session['recordings'])):
                    rec = op.join(session['absolute_path'], recording)
                    print(rec)
                    try:
                        video_ts, pupil = load_pupil_data(rec)
                    except Exception:
                        traceback.print_exc()
                        print('Something missing in this segment')
                        pupil = []
                    all_pupil.extend(pupil)
                    animal_pupil.extend(pupil)
        #pdb.set_trace()
        ax = axar[i]
        animal_pupil = np.asarray(animal_pupil)
        animal_pupil = animal_pupil[~np.isnan(animal_pupil)]
        hist, edges = np.histogram(animal_pupil)
        ax.bar(edges[:-1], hist, width=0.2,
               color=3*[.5],
               edgecolor='none',
               linewidth=0,
               align='edge')
        #ax.hist(animal_pupil)
        #mark the median
        ax.axvline(np.median(animal_pupil), color =  'k', linestyle = '--')
        ax.set_xlabel('Pupil (pxls)')
        ax.set_ylabel('# time-points')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(animal.split('/')[-2])
    ax = axar[4]
    all_pupil = np.asarray(all_pupil)
    all_pupil = all_pupil[~np.isnan(all_pupil)]
    ax.hist(all_pupil)
    ax.set_xlabel('Pupil (pxls)')
    ax.set_ylabel('# time-points')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.axvline(np.median(all_pupil), color='k', linestyle='--')
    ax.set_title('All Animals')
    filename = 'overall_pupil_histograms'
    fig.set_size_inches(5, 12)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.97, wspace=0.1, hspace=0.2)
    fig.tight_layout()
    fig.savefig(op.join(output, filename + '.pdf'))

cli.add_command(cli_overall_pupil)


@click.command(name='overall-running')
@click.argument('database', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--show', '-s', is_flag=True)
def cli_overall_running(database=None,
                    output=None,
                    show=False,
                    **kwargs):
    #code to create 2d histograms of the relationship between running and pupil for each animal and for all animals; plotted on the same figure
    animals = [op.join(database, animal, 'neural') for animal in os.listdir(database) if animal.startswith('M102') and not animal.endswith('_neural')]
    fig, axar = plt.subplots(nrows=5, ncols=1, sharex=True)
    all_data = []
    for i, animal in enumerate(sorted(animals)):
        animal_data = []
        #find sessions with drc segments
        sessions = helpers.get_sessions_with_drc_recordings(animal)
        for session in sorted(sessions):
            session = sessions[session]
            # find the drc segments
            if len(session['recordings']) > 0:
                for j, recording in enumerate(sorted(session['recordings'])):
                    rec = op.join(session['absolute_path'], recording)
                    print(rec)
                    try:
                        video_ts, pupil = load_pupil_data(rec)
                        t, f, b, speed = load_cylinder_running_data(rec)
                        f1 = interpolate.interp1d(video_ts, pupil,
                                                  kind='linear',
                                                  bounds_error=False,
                                                  fill_value=np.NaN)

                        f2 = interpolate.interp1d(t, speed,
                                                  kind='linear',
                                                  bounds_error=False,
                                                  fill_value=0)

                        # create new time frame
                        newt = np.arange(0.10, 675., 0.20)
                        new_pupil = f1(newt)
                        new_speed = f2(newt)
                        for k in range(new_pupil.shape[0]):
                            animal_data.append((new_pupil[k], new_speed[k]))
                            all_data.append((new_pupil[k], new_speed[k]))
                    except Exception:
                        traceback.print_exc()
                        print('Something missing in this segment')
        #pdb.set_trace()
        ax = axar[i]
        pupil_animal = np.asarray([x[0] for x in animal_data])
        speed_animal = np.asarray([x[1] for x in animal_data])
        v = np.isnan(pupil_animal)
        #pdb.set_trace()
        ax.hist2d( speed_animal[~v], pupil_animal[~v], bins = 15)
        #ax.scatter( speed_animal[~v], pupil_animal[~v], s = 0.5)
        ax.set_xlabel('Pupil (pxls)')
        ax.set_ylabel('Running speed (cm/s)')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(animal.split('/')[-2])
    ax = axar[4]
    pupil_all = np.asarray([x[0] for x in all_data])
    speed_all = np.asarray([x[1] for x in all_data])
    v = np.isnan(pupil_all)
    # pdb.set_trace()
    ax.hist2d( speed_all[~v], pupil_all[~v], bins=15)
    #ax.scatter(speed_all[~v], pupil_all[~v], s=0.5)
    ax.set_xlabel('Pupil (pxls)')
    ax.set_ylabel('Running speed (cm/s)')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_title('All Animals')
    filename = 'overall_pupil_running_2d_hists'
    fig.set_size_inches(5, 12)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.97, wspace=0.1, hspace=0.2)
    fig.tight_layout()
    fig.savefig(op.join(output, filename + '.pdf'))
    plt.close()

cli.add_command(cli_overall_running)

@click.command(name='overall-pupil-running')
@click.argument('database', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--threshold', '-t', default = 1.0)
@click.option('--show', '-s', is_flag=True)
def cli_overall_pupil_running(database=None,
                    output=None,
                    threshold = 1.0,
                    show=False,
                    **kwargs):
    #code to create pupil distributions during running and not-running
    animals = [op.join(database, animal) for animal in os.listdir(database) if animal.startswith('M102') and not animal.endswith('_neural')]
    fig, axar = plt.subplots(nrows=5, ncols=1, sharex=True)
    all_data = []
    bout_length = 10.
    for i, animal in enumerate(sorted(animals)):
        animal_data = []
        #find sessions with drc segments
        sessions = helpers.get_sessions_with_drc_recordings(animal)
        for session in sorted(sessions):
            session = sessions[session]
            # find the drc segments
            if len(session['recordings']) > 0:
                for j, recording in enumerate(sorted(session['recordings'])):
                    rec = op.join(session['absolute_path'], recording)
                    print(rec)
                    try:
                        video_ts, pupil = load_pupil_data(rec)
                        t, f, b, speed = load_cylinder_running_data(rec)
                        f1 = interpolate.interp1d(video_ts, pupil,
                                                  kind='linear',
                                                  bounds_error=False,
                                                  fill_value=np.NaN)

                        f2 = interpolate.interp1d(t, speed,
                                                  kind='linear',
                                                  bounds_error=False,
                                                  fill_value=np.NaN)

                        # create new time frame
                        newt = np.arange(0.10, 675., 0.20)
                        new_pupil = f1(newt)
                        new_speed = f2(newt)
                        for k in range(new_pupil.shape[0]):
                            animal_data.append((new_pupil[k], new_speed[k]))
                            all_data.append((new_pupil[k], new_speed[k]))
                    except Exception:
                        traceback.print_exc()
                        print('Something missing in this segment')
        #pdb.set_trace()
        ax = axar[i]
        pupil_animal = np.asarray([x[0] for x in animal_data])
        speed_animal = np.asarray([x[1] for x in animal_data])

        big_t = np.arange(0.1, speed_animal.shape[0] * 0.2, 0.2)

        run = speed_animal >= 1.0
        for m, (t, sp) in enumerate(zip(big_t, speed_animal)):
            if sp < 1.0:
                pass
            elif sp >= 1.0:
                # get the indeces of the elements between t =t and t=t+10s
                indeces = np.where((big_t >= t) & (big_t < t + bout_length))
                run[indeces] = True

        v = ~np.isnan(pupil_animal) & run
        w = ~np.isnan(pupil_animal) & ~run

        ax.hist(pupil_animal[v], color = 'blue', alpha = 0.5, histtype = 'bar',ec = 'black')
        ax.hist(pupil_animal[w], color = 'orange', alpha = 0.5, histtype = 'bar',ec = 'black')
        ax.set_xlabel('Pupil (rel. eye width)')
        ax.set_ylabel('Timepoints')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(op.split(animal)[1])
    ax = axar[4]
    pupil_all = np.asarray([x[0] for x in all_data])
    speed_all = np.asarray([x[1] for x in all_data])
    big_t = np.arange(0.1, speed_all.shape[0] * 0.2, 0.2)

    run = speed_all >= 1.0
    for i, (t, sp) in enumerate(zip(big_t, speed_all)):
        if sp < 1.0:
            pass
        elif sp >= 1.0:
            # get the indeces of the elements between t =t and t=t+10s
            indeces = np.where((big_t >= t) & (big_t < t + bout_length))
            run[indeces] = True

    v = ~np.isnan(pupil_all) & run
    w = ~np.isnan(pupil_all) & ~run
    ax.hist(pupil_all[v],histtype = 'bar',ec = 'black', color='blue', alpha = 0.5, label = 'running bout')
    ax.hist(pupil_all[w],histtype = 'bar',ec = 'black', color='orange', alpha = 0.5, label = 'non running bout')
    ax.set_xlabel('Pupil (rel. eye width)')
    ax.set_ylabel('Timepoints')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axar.flat:
        right_side = ax.spines["right"]
        right_side.set_visible(False)
        top_side = ax.spines["top"]
        top_side.set_visible(False)
        helpers.adjust_spines(ax, ['bottom', 'left'])
    ax.set_title('All Animals')
    ax.legend()
    filename = 'overall_pupil_dists_run_norun_t_' + str(threshold)
    fig.set_size_inches(5, 12)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.97, wspace=0.1, hspace=0.2)
    fig.tight_layout()
    fig.savefig(op.join(output, filename + '.pdf'))
    plt.close()

cli.add_command(cli_overall_pupil_running)


@cli.command('running-bout-example')
@click.argument('rec_path', type=click.Path(exists=True))
def cli_running_bout_example(rec_path):

    from helpers import load_cylinder_running_data, extract_running_bouts

    threshold = 2  # running threshold in cm/s
    binwidth = .2
    smooth_len = 2.

    t, backward, forward, speed = load_cylinder_running_data(rec_path,
                                                             binwidth=binwidth,
                                                             t_range=None,
                                                             f_smoothing=0  # 0 disables smoothing inside function
                                                             )

    # use normalized window for smoothing to preserve scaling
    n = smooth_len / float(binwidth)
    if n % 2 == 0:
        # make sure window is symmetric about 0
        n += 1

    from scipy import signal
    win = signal.gaussian(n, .15*n)
    speed = np.convolve(speed, win / win.sum(), 'same')

    # extract running bouts using function defined in "helpers.py"
    on_off_times = extract_running_bouts(t, speed,
                                         threshold=threshold,  # running threshold in cm/s
                                         min_still=0,  # minimum still period before running start
                                         max_still=np.Inf,  # maximum still period before running start
                                         min_duration=1,  # minimum running bout diration
                                         verbose=True  # print number of extracted/ignored running bouts
                                         )

    fig, ax = plt.subplots()
    ax.plot(t, speed, '-', label='speed')
    ax.axhline(threshold, color='r', ls='--', lw=1, label='threshold')
    for i, (t1, t2) in enumerate(on_off_times):
        ax.axvspan(t1, t2, color=3*[.5], ec='none', label='running bout' if i == 0 else None)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Running speed (cm/s)')

    ax.legend(loc='best').get_frame().set_linewidth(0)
    fig.tight_layout()

    plt.show(block=True)


if __name__ == '__main__':
    cli()

