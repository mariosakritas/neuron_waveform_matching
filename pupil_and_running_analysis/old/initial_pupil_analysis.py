from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path as op
import sys
from scipy import signal
from scipy import interpolate
from scipy import stats
import statsmodels.stats.multitest as multi


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
    speed = np.sqrt(backward ** 2 + forward ** 2)

    return t, backward, forward, speed


def get_color_condition(cond):

    colors = {'Rand': 3*[.4],
              'Const': 3*[.7],
              'RegRand': '#1f77b4',
              'Step': '#ff7f0e',
              'RandReg': '#bcbd22',
              'Reg': '#e377c2'}

    return colors[cond]


def run_example(rec_path):

    # load pupil data
    pupil_file = op.join(rec_path, 'rpi_camera_2_pupil_data.npz')
    pupil_data = np.load(pupil_file, encoding='latin1')

    pupil_size = np.max(pupil_data['size'], axis=1)

    # smooth pupil trace (needs more work)
    win = signal.gaussian(11, 1.5)
    pupil_size = np.convolve(pupil_size, win/win.sum(), 'same')

    # camera timestamps; should be one per frame but sometimes the last few
    # triggers were not received by open-ephys
    cam = np.load(op.join(rec_path, 'rpi_camera_2.npz'), encoding='latin1')
    ts = cam['timestamps']

    N = min(pupil_size.shape[0], ts.shape[0])
    ts = ts[:N]
    pupil_size = pupil_size[:N]

    # load stimulus events
    event_data = np.load(op.join(rec_path, 'stim_events.npz'), encoding='latin1')
    stim_events = event_data['StimEvents'].tolist()

    # # plot trace
    # fig, ax = plt.subplots()
    #
    # ax.plot(ts, pupil_size, '-', color=3*[.1])
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Pupil size (pixels)')
    #
    # conditions = list(set([ev['Stim'] for ev in stim_events]))
    # print("conditions:", conditions)
    #
    # for i, ev in enumerate(stim_events):
    #
    #     t0 = ev['Timestamp']
    #     T = ev['Duration']
    #
    #     cond = ev['Stim']
    #     c = get_color_condition(cond)
    #
    #     ax.axvspan(t0, t0+T, lw=0,
    #                facecolor=c, alpha=.5)
    #
    #     t_transition = ev['Transition']
    #     if t_transition > 0:
    #         ax.axvline(t0 + t_transition, color=c, lw=1)
    #
    t, backward, forward, speed = load_cylinder_running_data(rec_path)
    # ax.plot(t, speed, 'b-', linewidth=0.3, label='Speed')
    # ax.legend(loc='best')
    #
    # fig.tight_layout()
    # plt.show()


    # create function for interpolation for pupil size
    f1 = interpolate.interp1d(ts, pupil_size,
                              kind='linear',
                              bounds_error=False,
                              fill_value=np.NaN)

    # interpolation function for speed data
    f2 = interpolate.interp1d(t, speed,
                              kind='linear',
                              bounds_error=False,
                              fill_value=np.NaN)

    new_time = np.arange(0,speed.shape[0],1)
    new_pupil = f1(new_time)
    new_speed = f2(new_time)

    mask = ~np.isnan(new_speed) & ~np.isnan(new_pupil)
    print(new_speed[mask].shape)
    print(new_pupil[mask].shape)

    slope, intercept, r_value, p_value, std_err = stats.linregress(new_speed[mask], new_pupil[mask])
    print (slope, intercept, r_value, p_value, std_err)

    # # make scatter plot
    # fig2, ax2 = plt.subplots()
    # ax2.scatter(new_speed, new_pupil, s=3., c='k')
    # ax2.set_xlabel('Speed (cm/s)')
    # ax2.set_ylabel('Pupil size (pixels)')
    # xi = np.arange(0,12,1)
    # line = slope*xi +intercept
    # ax2.plot(xi,line,'r-', label='r = 0.22 *')
    # ax2.legend(loc=(.75,.62), fancybox= True, framealpha=0.5)
    # plt.show()

    pup_speedy=[]
    pup_not_so_speedy=[]
    #make histogram
    for sp, pup in zip(new_speed, new_pupil):
        if sp < 1.0:
            pup_not_so_speedy.append(pup)
        else:
            pup_speedy.append(pup)

    pup_speedy = np.asarray(pup_speedy)
    pup_not_so_speedy = np.asarray(pup_not_so_speedy)
    mask1 = ~np.isnan(pup_speedy)
    mask2 = ~np.isnan(pup_not_so_speedy)
    # my_sum = (pup_speedy[mask1].shape[0]) + (pup_not_so_speedy[mask2].shape[0])
    # print(my_sum)
    #

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)
    fig3, ax3 = plt.subplots()
    ax3.hist(pup_speedy[mask1], bins=50, color='blue', label='Running', alpha=0.7)
    ax3.hist(pup_not_so_speedy[mask2], bins=50, color='fuchsia', label= 'Still', alpha =0.8)
    ax3.set_xlabel('Pupil size (pixels)')
    ax3.set_ylabel('Frequency')


    ax3.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.show()

if __name__ == '__main__':

    rec_path = '/Users/user/Desktop/PhD/2ndrot/randreg_database/M10205/2018_05_26/reg2/2018-05-26_11-32-04_ChaitStimulus' \
               ''
    if len(sys.argv) > 1:
        rec_path = sys.argv[1]

    run_example(rec_path)
