# Author: Marios Akritas

"""
analysis for 2AFC expts
"""

from __future__ import print_function

import click
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from scipy import stats
import traceback

#import stimcontrol.util as scu

from ipdb import set_trace as db

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt



def filter_data(x, fs, f_cutoff=5., order=2):

    Wn = f_cutoff / fs * 2
    b, a = signal.butter(order, Wn,
                         btype='lowpass',
                         analog=False,
                         output='ba')

    data = signal.filtfilt(b, a, x, axis=0)

    return data


def load_reward_times(rec_path):

    dd = np.load(op.join(rec_path, 'ttl_events.npz'), encoding='latin1')

    v = np.logical_and(dd['channels'] == 7, dd['states'] > 0)
    ts = dd['timestamps'][v]

    return ts


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


def create_filename_from_session_path(session_path):

    parts = session_path.split(op.sep)

    return parts[-4] + '_' + parts[-2] + '_' + parts[-1]



#   first we need a trial analysis: plot the analogue signal from the 3 lickspouts on 3 separate plots and
#   include the start of the waiting period (when lamp switches off).
#   and the time of reward (variable depending on the delay: start_of_delay + t_delay)
#

# load events

def analyse_trials(session_dir):

    t_pre = .5
    t_post = .5

    #load events
    stim_events = np.load(op.join(session_dir , 'stim_events.npz'), encoding='latin1')['StimEvents']
    start_events = [ev for ev in stim_events if ev['EventType'] == 'TrialStart']
    print(stim_events)
    #load analogue signal
    adc = np.load(op.join(session_dir , 'adc_channels.npz'), encoding='latin1')
    adc_channels = adc['data']  #shape: 6897783.3
    adc_ts = adc['timestamps'] #shape: 6897783.1
    n_channels = adc_channels.shape[1]

    # z-score channels
    # for j in range(n_channels):
    #     adc_channels[:, j] = adc_channels[:, j] / np.std(adc_channels[:, j])

    # --- load licking events ---
    lick_list = []
    lickev_file_left = op.join(session_dir, 'crossing_detector_adc_1_ttl_2.npz')
    lickev_file_centre = op.join(session_dir, 'crossing_detector_adc_2_ttl_3.npz')
    lickev_file_right = op.join(session_dir, 'crossing_detector_adc_3_ttl_4.npz')
    lickev_ts = None
    if op.exists(lickev_file_left):
        lickev_data_left = np.load(lickev_file_left, encoding='latin1')
        lickev_ts_left = lickev_data_left['timestamps']
        states = lickev_data_left['channel_states']
        lickev_ts_left = lickev_ts_left[states > 0]
        lick_list.append(lickev_ts_left)
    if op.exists(lickev_file_centre):
        lickev_data_centre = np.load(lickev_file_centre, encoding='latin1')
        lickev_ts_centre = lickev_data_centre['timestamps']
        states = lickev_data_centre['channel_states']
        lickev_ts_centre = lickev_ts_centre[states > 0]
        lick_list.append(lickev_ts_centre)
    if op.exists(lickev_file_right):
        lickev_data_right = np.load(lickev_file_right, encoding='latin1')
        lickev_ts_right = lickev_data_right['timestamps']
        states = lickev_data_right['channel_states']
        lickev_ts_right = lickev_ts_right[states > 0]
        lick_list.append(lickev_ts_right)

    events_trials = []
    i = 0
    while i < len(stim_events):

        ev = stim_events[i]
        if ev['EventType'] == 'TrialStart':

            events_trial = {'TrialStart': ev}
            i += 1
            while i < len(stim_events):

                ev = stim_events[i]
                if ev['EventType'] != 'TrialStart':
                    events_trial[ev['EventType']] = ev
                    i += 1
                else:
                    events_trial['TrialStop'] = ev
                    break
            events_trials.append(events_trial)
        else:
            i += 1

    trial_pdf = PdfPages(op.join(session_dir, 'trial_analysis.pdf'))

    for i, events_trial in enumerate(events_trials):

        #print some trial information
        print(50*'-')
        print('Trial {}/{}'.format(i+1, len(events_trials)))

        for k in events_trial:
            if len(events_trial) > 4:
                ev = events_trial[k]
                tst = ev['Timestamp']
                print("  ", k, tst) #, ev)


        # get trial start/stop times
        start_ev = events_trial['RewardDelay']
        t_start = start_ev['Timestamp']
        if 'TrialStop' in events_trial:
            t_stop = events_trial['TrialStop']['Timestamp']
        else:
            # TODO: get total length of recording ...
            t_stop = adc_ts.max()

        fig, axes = plt.subplots(nrows=n_channels,
                                 ncols=1,
                                 sharex=True,
                                 sharey=True)

        # extract analog signal for trial
        v = np.logical_and(adc_ts >= t_start - t_pre,
                           adc_ts <= t_stop + t_post)

        order = ['Left', 'Centre', 'Right']

        for j in range(n_channels):

            x = adc_channels[v, j]
            ax = axes[j]
            ax.plot(adc_ts[v] - t_start, x, '-', color=3*[.1], lw = 0.1)

            licks = lick_list[j]
            g = np.logical_and(licks >= t_start - t_pre,
                               licks <= t_stop + t_post)
            for k in range(len(licks[g])):
                ax.axvline(licks[g][k] - t_start, color= 3*[.3])

            ax.set_xlabel('Time (s)')
            ax.set_ylabel(order[j])

            # add events
            if 'RewardDelay' in events_trial:
                ev = events_trial['RewardDelay']
                t0 = ev['Timestamp']
                t_delay = ev['RewardDelay']
                ax.axvspan(t0 - t_start, t0 - t_start + t_delay,
                           facecolor=3*[.5],
                           alpha=.5,
                           lw=0)
            if 'PunishmentTimeout' in events_trial:
                ev = events_trial['PunishmentTimeout']
                t0 = ev['Timestamp']
                t_timeout = ev['PunishmentTimeout']
                ax.axvline(t0 - t_start,
                           color='r')
            if 'RewardSound' in events_trial:
                ev = events_trial['RewardSound']
                t0 = ev['Timestamp']
                ax.axvline(t0 - t_start,
                           color='g')
            if 'LickReward' in events_trial:
                ev = events_trial['LickReward']
                t0 = ev['Timestamp']
                ax.axvline(t0 - t_start,
                           color='b')
            if 'TrialStop' in events_trial:
                ev = events_trial['TrialStop']
                t0 = ev['Timestamp']
                ax.axvline(t0 - t_start, linestyle='--',
                           color='k', lw=2)

            ax.set_xlim(-t_pre, t_stop-t_start + t_post)

        # SAVE PLOT ONTO pdf pages thingy
        #plt.title('Page' + str(i+1), loc= 'lower right')
    #     trial_pdf.savefig()
    # trial_pdf.close()
    #

analyse_trials('/Users/user/Desktop/PhD/MAIN/database_no_videos/M10231/2019_05_28/2AFC/2019-05-28_12-44-50_TwoAFCStage2')



def get_correct_trials(session_dir):

    print('analysing:', session_dir)
    #load events
    stim_events = np.load(op.join(session_dir , 'stim_events.npz'), encoding='latin1')['StimEvents']
    start_events = [ev for ev in stim_events if ev['EventType'] == 'TrialStart']

    events_trials = []
    i = 0
    while i < len(stim_events):

        ev = stim_events[i]
        if ev['EventType'] == 'TrialStart':

            events_trial = {'TrialStart': ev}
            i += 1
            while i < len(stim_events):

                ev = stim_events[i]
                if ev['EventType'] != 'TrialStart':
                    events_trial[ev['EventType']] = ev
                    i += 1
                else:
                    events_trial['TrialStop'] = ev
                    break
            events_trials.append(events_trial)
        else:
            i += 1

    correct_trials = 0

    for i, events_trial in enumerate(events_trials):
        for k in events_trial:
            if k == 'LickReward':
                correct_trials += 1
    try:
        percent_correct = correct_trials*100 / len(events_trials)
    except:
        percent_correct = float('nan')
        
    return percent_correct


def get_correct_trials_across_sessions(db_path):
    session_paths = []
    for root, dirs, files in os.walk(db_path):
        for d in dirs:
            if 'TwoAFCStage2' in d:
                session_paths.append(op.join(root, d))

    session_paths = sorted(session_paths)

    correct_trials_across = []
    for session_path in session_paths:
        correct = get_correct_trials(session_path)
        correct_trials_across.append(correct)
        print("analyzing session:", session_path, 'correct', correct)

    

    fig, ax = plt.subplots()
    ax.plot(range(0,len(session_paths),1), correct_trials_across)
    ax.set_xlabel('Session')
    ax.set_ylabel('% of correct trials')
    plt.title('2AFC Stage 2 - '+op.split(db_path)[-1])
    plt.show()



#get_correct_trials_across_sessions('/Users/user/Desktop/PhD/MAIN/database_no_videos/M10233')
