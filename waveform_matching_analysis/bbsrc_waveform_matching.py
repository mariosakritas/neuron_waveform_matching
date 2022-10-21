#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit (waveform) tracking across sessions using the
algorithm described in

Tolias, A. S.; Ecker, A. S.; Siapas, A. G.; Hoenselaar, A.; Keliris, G. A. & Logothetis, N. K.
Recording chronically from the same neurons in awake, behaving primates.
Journal of neurophysiology, 2007, 98, 3780-3790

@author: arne, jules, marios
"""


from __future__ import print_function

import os.path as op
import click
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import itertools
import math
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import make_spd_matrix
from scipy.stats import multivariate_normal
from scipy.stats import chi2
from datetime import datetime
from collections import Counter

from stimcontrol.io.database import NeoIO
from stimcontrol.util import makedirs_save
import stimcontrol.util as scu

import helpers
import pdb

# from ipdb import set_trace as db


@click.group()
def cli():
    pass


def get_days_between_dates(d1, d2):

    d1 = datetime.strptime(d1, "%Y_%m_%d")
    d2 = datetime.strptime(d2, "%Y_%m_%d")

    return abs((d2 - d1).days)


def get_days_between_units(u1, u2):
    day1 = u1['date']
    day2 = u2['date']
    days = get_days_between_dates(day1, day2)
    
    return days


def get_waveforms_sessions(path, num_tetrodes=8):

    sessions = helpers.get_sessions_with_drc_recordings(path)

    tetrode_depths = []
    days_after_first_session = []
    date_first_session = None
    session_names = []

    waveforms_tetrodes = {k+1: [] for k in range(num_tetrodes)}
    waveform_units = {k+1: [] for k in range(num_tetrodes)}

    for s in sorted(sessions):

        print(s)
        print("  ", sessions[s]['absolute_path'])
        print("  ", sessions[s]['recordings'])

        session = sessions[s]
        session_names.append(sessions[s]['relative_path'])

        # ----- get tetrode depth (relative to initial position) -----
        advancer_data = np.load(op.join(session['absolute_path'], 'advancer_data.npz'),
                                encoding='latin1',
                                allow_pickle=True)
        dd = advancer_data['channelgroup_1'].item()
        tetrode_depths.append(dd['Depth'])

        if date_first_session is None:
            days = 0
            date_first_session = session['date']
        else:
            days = get_days_between_dates(date_first_session, session['date'])

        days_after_first_session.append(days)

        # ----- get waveforms -----
        reader = NeoIO(session['absolute_path'])
        block = reader.read_block(recordings=session['recordings'],
                                  dtypes=['spikes'],
                                  cluster_groups=['Good', 'MUA'])
        units = block.list_units

        print("    segments:", len(block.segments))
        print("    units:", len(units))

        for i in range(num_tetrodes):

            waveforms = []
            wf_units = []
            for unit in units:
                # extract waveforms here
                if unit.annotations['channel_index'] == i+1:
                    waveforms.append(unit.annotations['waveform']['mean'])
                    wf_units.append(unit.annotations['unit_index'])

            waveforms_tetrodes[i+1].append(waveforms)
            waveform_units[i+1].append(wf_units)

    return waveforms_tetrodes, tetrode_depths, days_after_first_session, session_names


def load_spike_units(path):

    sessions = helpers.get_sessions_with_drc_recordings(path)

    data = []
    date_first_session = None
    index = 0

    for s in sorted(sessions):

        session = sessions[s]

        print(s)
        print("  ", session['absolute_path'])
        print("  ", session['recordings'])

        # ----- get tetrode depth (relative to initial position) -----
        advancer_data = np.load(op.join(session['absolute_path'], 'advancer_data.npz'),
                                encoding='latin1',
                                allow_pickle=True)
        dd = advancer_data['channelgroup_1'].item()
        depth = dd['Depth']

        if date_first_session is None:
            days = 0
            date_first_session = session['date']
        else:
            days = get_days_between_dates(date_first_session, session['date'])

        # ----- get waveforms -----
        reader = NeoIO(session['absolute_path'])
        block = reader.read_block(recordings=session['recordings'],
                                  dtypes=['spikes'],
                                  cluster_groups=['Good', 'MUA'])
        units = block.list_units

        print("    segments:", len(block.segments))
        print("    units:", len(units))

        for unit in units:

            data.append(helpers.SpikeUnit(tetrode=unit.annotations['channel_index'],
                                          cluster=unit.annotations['unit_index'],
                                          waveforms=unit.annotations['waveform']['mean'],
                                          cluster_group=unit.annotations['group'],
                                          index=index,
                                          depth=depth,
                                          session=sessions[s]['relative_path'],
                                          recording_day=days,
                                          date=session['date']))
            index += 1

    return data


def get_days_between_recordings(path, count_recordings=True):
    """compute the frequency of days between recordings

        Compute frequency of days between recording sessions by
        taking into account electrode depth, i.e. only compare
        sessions for the same depth. Moreover, get the number of
        DRC recordings (within sessions) which is helpful for
        computing the number of units per recording day
        difference (done elsewhere).

    """

    sessions = helpers.get_sessions_with_drc_recordings(path)

    tetrode_depths = []
    days_after_first_session = []
    recordings_per_session = []
    date_first_session = None

    for s in sorted(sessions):

        print(50*'-')
        print(s)
        print("  ", sessions[s]['absolute_path'])
        print("  ", sessions[s]['recordings'])

        session = sessions[s]

        advancer_data = np.load(op.join(session['absolute_path'], 'advancer_data.npz'),
                                encoding='latin1',
                                allow_pickle=True)
        dd = advancer_data['channelgroup_1'].item()
        tetrode_depths.append(dd['Depth'])

        if date_first_session is None:
            days = 0
            date_first_session = session['date']
        else:
            days = get_days_between_dates(date_first_session, session['date'])

        days_after_first_session.append(days)
        recordings_per_session.append(len(session['recordings']))

        print("  date:", session['date'])
        print("  depth:", dd['Depth'])

    tetrode_depths = np.asarray(tetrode_depths)
    days_after_first_session = np.asarray(days_after_first_session)
    recordings_per_session = np.asarray(recordings_per_session)

    results = []
    for depth in np.unique(tetrode_depths):

        ind = np.where(tetrode_depths == depth)[0]

        # test all unique combinations (i.e. do not count combinations twice)
        for i1 in ind:
            for i2 in ind:
                if i2 > i1:

                    days = days_after_first_session[i2] - days_after_first_session[i1]

                    if count_recordings:
                        n = recordings_per_session[i1] * recordings_per_session[i2]
                        results.append(np.repeat(days, n))
                    else:
                        results.append(days)

    results = np.concatenate(results)

    return results


def remove_multiple_matched_day(unit_indices, 
                                thresholded_indices,
                                dist_shape_scale,
                                r_ellipse_null):
    ''' Some units were matched with units from the same day.
    This function specifically targets these previously identified units.
    It calculates the distance between d1, d2 of the pairs of units of interest
    and the null ellipse. Then, it only keeps the unit that is farthest from
    the null. Code could be optimized'''


    # duplicate with 133, 134 and 135:
    p133 = np.array([pair for pair in thresholded_indices if 133 in pair])
    indices133 = [i for i, p in enumerate(unit_indices) 
                  if (p == p133).all(axis = 1).any()]

    p134 = np.array([pair for pair in thresholded_indices if 134 in pair])
    indices134 = [i for i, p in enumerate(unit_indices) 
                  if (p == p134).all(axis = 1).any()]

    
    p135 = np.array([pair for pair in thresholded_indices if 135 in pair])
    indices135 = [i for i, p in enumerate(unit_indices) 
                  if (p == p135).all(axis = 1).any()]
            
    d133 = [dist_shape_scale[i] for i in indices133]
    d134 = [dist_shape_scale[i] for i in indices134]
    d135 = [dist_shape_scale[i] for i in indices135]

    distances = [d133, d134, d135]
    means = []
    for i, cell in enumerate(distances):
        d = []
        for pair in cell:
            # Calculate the smallest distance from the null ellipse
            dist = [np.sqrt( (ell[0] - pair[0])**2 + (ell[1] - pair[1])**2) 
                    for ell in r_ellipse_null]
            d.append(min(dist))
        means.append(np.mean(d))
        
    if np.max(means) == means[0]:
        remove_indices = np.concatenate((p134, p135))
    elif np.max(means) == means[1]:
        remove_indices = np.concatenate((p133, p135))
    elif np.max(means) == means[2]:
        remove_indices = np.concatenate((p133, p134)) 
        
    # duplicate with 164 and 165:
        
    p164 = np.array([pair for pair in thresholded_indices if 164 in pair])
    indices164 = [i for i, p in enumerate(unit_indices) 
                  if (p == p164).all(axis = 1).any()]

    p165 = np.array([pair for pair in thresholded_indices if 165 in pair])
    indices165 = [i for i, p in enumerate(unit_indices) 
                  if (p == p165).all(axis = 1).any()]
    
    d164 = [dist_shape_scale[i] for i in indices164]
    d165 = [dist_shape_scale[i] for i in indices165] 
    
    distances = [d164, d165]
    means = []
    for i, cell in enumerate(distances):
        d = []
        for pair in cell:
            # Calculate the smallest distance from the null ellipse
            dist = [np.sqrt( (ell[0] - pair[0])**2 + (ell[1] - pair[1])**2) 
                    for ell in r_ellipse_null]
            d.append(min(dist))
        means.append(np.mean(d))

    if np.max(means) == means[0]:
        remove_indices = np.concatenate((remove_indices, p165))
    elif np.max(means) == means[1]:
        remove_indices = np.concatenate((remove_indices, p164))
        
    # Remove units from thresholded indices:
        
    new_thresholded_indices = np.array([pair for pair in thresholded_indices
                        if not (pair == remove_indices).all(axis = 1).any()])
    
    return(new_thresholded_indices)

# -----------------------------------------------------------------------------
# plot some measures for a single session
# -----------------------------------------------------------------------------
@click.command(name='session')
@click.argument('session_path', type=click.Path(exists=True))
def cli_session(session_path=None):

    recordings = helpers.get_recordings(session_path,
                                        patterns=['_DRC'])
    reader = NeoIO(session_path)
    block = reader.read_block(recordings=recordings,
                              dtypes=['spikes'],
                              cluster_groups=['Good', 'MUA'])
    units = block.list_units
    n_units = len(units)

    print("    segments:", len(block.segments))
    print("    units:", len(units))

    if n_units > 0:

        channels = np.array([u.annotations['channel_index'] for u in units])

        for c in np.unique(channels):

            units_chan = [u for u in units if u.annotations['channel_index'] == c]

            if len(units_chan) > 0:

                # plot waveforms
                fig, axes = plt.subplots(nrows=1,
                                         ncols=len(units_chan),
                                         sharex=True,
                                         sharey=True)
                axes = np.atleast_1d(axes)

                for i, unit in enumerate(units_chan):

                    W = unit.annotations['waveform']['mean']
                    ax = axes[i]
                    ax.plot(W)

                fig.tight_layout()

                # covariance matrices (or rather: matrix products)
                fig, axes = plt.subplots(nrows=max(len(units_chan), 2),
                                         ncols=max(len(units_chan), 2),
                                         sharex=True,
                                         sharey=True)
                for i, unit1 in enumerate(units_chan):

                    W1 = unit1.annotations['waveform']['mean']

                    for j, unit2 in enumerate(units_chan):

                        W2 = unit2.annotations['waveform']['mean']
                        c = np.dot(W1.T, W2)

                        ax = axes[i, j]
                        ax.imshow(c)

                fig.tight_layout()

    plt.show(block=True)


cli.add_command(cli_session)


# -----------------------------------------------------------------------------
# plot tetrode depth as a function of days after the first session
# -----------------------------------------------------------------------------
@click.command(name='depths')
@click.argument('path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None)
def cli_depths(path=None,
               output=None):

    sessions = helpers.get_sessions_with_drc_recordings(path)

    tetrode_depths = []
    days_after_first_session = []
    date_first_session = None

    for s in sorted(sessions):

        print(50*'-')
        print(s)
        print("  ", sessions[s]['absolute_path'])
        print("  ", sessions[s]['recordings'])

        session = sessions[s]

        advancer_data = np.load(op.join(session['absolute_path'], 'advancer_data.npz'),
                                encoding='latin1',
                                allow_pickle=True)
        dd = advancer_data['channelgroup_1'].item()
        tetrode_depths.append(dd['Depth'])

        if date_first_session is None:
            days = 0
            date_first_session = session['date']
        else:
            days = get_days_between_dates(date_first_session, session['date'])

        days_after_first_session.append(days)

        print("  date:", session['date'])
        print("  depth:", dd['Depth'])

    tetrode_depths = np.asarray(tetrode_depths)
    days_after_first_session = np.array(days_after_first_session)

    fig, ax = plt.subplots()
    ax.plot(days_after_first_session, tetrode_depths, '-',
            lw=1.,
            color=3*[.5],
            alpha=.5)
    ax.plot(days_after_first_session, tetrode_depths, 'o',
            color=3*[.1],
            mec='none',
            ms=2.5)

    ax.set_xlabel('Days after first session')
    ax.set_ylabel('Tetrode depth (um)')
    ax.set_xlim(days_after_first_session[0]-1, days_after_first_session[-1]+1)
    ax.set_ylim(0, np.max(tetrode_depths)+50)

    scu.set_font_axes(ax)
    scu.simple_xy_axes(ax)
    scu.adjust_axes(ax, pad=2)

    fig.set_size_inches(5, 2.5)
    fig.tight_layout()

    if output is not None:

        output = op.realpath(op.expanduser(output))
        makedirs_save(output)

        for ff in ['png', 'pdf']:
            fig.savefig(op.join(output, 'waveform_matching_tetrode_depths.' + ff),
                        format=ff,
                        dpi=300)

    plt.show(block=True)


cli.add_command(cli_depths)


# -----------------------------------------------------------------------------
# spike waveform matching across sessions (based on Tolia et al. 2007)
# -----------------------------------------------------------------------------

def vecnorm(v):
    return np.sqrt(np.sum(v ** 2))


def compute_waveform_distance(x, y):
    # compute waveform distance (d_1: shape, d_2: scaling)

    alphas = np.zeros((4,))
    for i in range(4):
        # p. 3784
        # For each channel, for a pair of average waveforms x and y, we
        # first scale x by alpha to minimize the sum of squared
        # differences between x and y. We refer to the scaling factor
        # of x and y as alpha(x, y).
        alphas[i] = np.dot(x[:, i], y[:, i]) / np.dot(x[:, i], x[:, i])

    # We then compute two different distance measures d_1 and d_2.
    # d_1 is a normalized Euclidean distance between the scaled
    # waveforms where the sum is over the four channels i. This solely
    # captures the difference in shape because the x_i values have been
    # scaled to match y_i and both have further been scaled by vecnorm(y_i).
    d_1 = 0.
    for i in range(4):
        d_1 += vecnorm(alphas[i] * x[:, i] - y[:, i]) / vecnorm(y[:, i])

    # d_2 captures the difference in amplitudes across the four channels.
    d_2a = np.max(np.abs(np.log(alphas)))
    d_2b = []
    for i in range(4):
        for j in range(4):
            d_2b.append(np.abs(np.log(alphas[i]) - np.log(alphas[j])))
    d_2 = d_2a + max(d_2b)

    return np.array([d_1, d_2])


def compute_waveform_distances_between_units(spike_units):

    dist_shape_scale = []
    unit_indices = []
    days_between_units = []

    # iterate over tetrodes (and then depth etc)
    for tetrode in np.unique([u.tetrode for u in spike_units]):
        # get all units on this tetrodes (across all sessions, depths etc)
        units_tetrode = [u for u in spike_units if u.tetrode == tetrode]
        for depth in np.unique([u.depth for u in units_tetrode]):
            # get all all units for this depth
            units_depth = [u for u in units_tetrode if u.depth == depth]
            # sort units by recording day (= session)
            recording_days_units = [u.recording_day for u in units_depth]
            recording_days = np.unique(recording_days_units)
            if len(recording_days) > 1:
                # compute waveform distance across recording days (= across sessions)
                for d in recording_days[:-1]:
                    units_day = [u for u in units_depth if u.recording_day == d]
                    units_other_days = [u for u in units_depth if u.recording_day > d]
                    for u1 in units_day:
                        x = u1.waveforms
                        date1 = u1.date
                        for u2 in units_other_days:
                            y = u2.waveforms
                            date2 = u2.date
                            # The final distance function is a vector-valued
                            # function of the single distance measures
                            d = compute_waveform_distance(x, y) + compute_waveform_distance(y, x)
                            day_diff = get_days_between_dates(date1, date2)
                            dist_shape_scale.append(d)
                            unit_indices.append((u1.index, u2.index))
                            days_between_units.append(day_diff)

    return np.vstack(dist_shape_scale), np.asarray(unit_indices), days_between_units

def compute_waveform_distances_null(spike_units, nb_q_turn):
    
    q_turn_depth = 62
    dist_shape_scale = []
    unit_indices = []

    # iterate over tetrodes (and then depth etc)
    for tetrode in np.unique([u.tetrode for u in spike_units]):

        # get all units on this tetrodes (across all sessions, depths etc)
        units_tetrode = [u for u in spike_units if u.tetrode == tetrode]
        
        # sort units by recording depth
        recording_depth = np.unique([u.depth for u in units_tetrode])

        if len(recording_depth)>1 and recording_depth[-1] - recording_depth[0] > q_turn_depth*nb_q_turn:
                
            for d in recording_depth[:-1]:
                
                units_depth = [u for u in units_tetrode if u.depth == d]
                units_other_depths = [u for u in units_tetrode 
                                      if u.depth >= d + (nb_q_turn * q_turn_depth)]
                
                if len(units_other_depths) > 0:
                    
                    for u1 in units_depth:
                        
                        x = u1.waveforms
                        
                        for u2 in units_other_depths:
                            
                        
                            y = u2.waveforms
                            # The final distance function is a vector-valued
                            # function of the single distance measures
                            d = compute_waveform_distance(x, y) + compute_waveform_distance(y, x)
                            dist_shape_scale.append(d)
                            unit_indices.append((u1.index, u2.index))

    if len(dist_shape_scale) > 0:        
        return np.vstack(dist_shape_scale), np.asarray(unit_indices)
                            
    else:
        return [], []


def compute_waveform_distances_old_null(spike_units1, spike_units2):
    dist_shape_scale = []
    unit_indices = []
    
    for u1 in spike_units1:
        
        x = u1.waveforms
        
        for u2 in spike_units2:
            
            y = u2.waveforms
            
            d = compute_waveform_distance(x, y) + compute_waveform_distance(y, x)
            dist_shape_scale.append(d)
            unit_indices.append((u1.index, u2.index))
    
    return np.vstack(dist_shape_scale), np.asarray(unit_indices)



def get_ellipse(mean, covariances, confidence=0.99, nb_points=1000):
    # Calculate the eigenvectors and eigenvalues
    eigenval, eigenvec = np.linalg.eigh(covariances)

    # Get the index of the largest eigenvector
    largest_eigenvec_ind = np.where(eigenval == np.max(eigenval))[0][0]
    largest_eigenvec = eigenvec[largest_eigenvec_ind]

    # Get the largest eigenvalue
    largest_eigenval = np.max(eigenval)

    # Get the smallest eigenvector and eigenvalue
    smallest_eigenval = np.min(eigenval)
    smallest_eigenvec = eigenvec[np.where(eigenval == smallest_eigenval)[0][0]]

    # Calculate the angle between the x-axis and the largest eigenvector
    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])

    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    if angle < 0:
        angle = angle + 2*np.pi

    # Get the 95% confidence interval error ellipse
    chisquare_val = np.sqrt(chi2.ppf(q = confidence, df = 2))
    theta_grid = np.linspace(0, 2*np.pi, nb_points)
    phi = angle
    X0 = mean[0]
    Y0 = mean[1]
    a = chisquare_val * np.sqrt(largest_eigenval)
    b = chisquare_val * np.sqrt(smallest_eigenval)

    # The ellipse in c and y coordinates
    ellipse_x_r = a*np.cos(theta_grid)
    ellipse_y_r = b*np.sin(theta_grid)

    # Define a rotation matrix
    R = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])

    # Let's rotate the ellipse to some angle phi
    r_ellipse = np.dot(np.array([ellipse_x_r, ellipse_y_r]).T, R)
    r_ellipse[:,0] = r_ellipse[:,0] + X0
    r_ellipse[:,1] = r_ellipse[:,1] + Y0
    
    ellipse_object = mpl.patches.Ellipse([X0, Y0], width=2*b, height=2*a, angle=90 + math.degrees(phi))
    
    return(r_ellipse, ellipse_object)
    
    
@click.command(name='tolias2007')
@click.argument('path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None)
@click.option('--overwrite', '-O', is_flag=True)
@click.option('--create-pdf', '-c', is_flag=True)
def cli_tolias2007(path=None,
                   output=None,
                   overwrite=False,
                   create_pdf=False):
    '''
    # waveform matching based on Tolias et al. J Neurophysiol 2007

    # user-defined thresholds for shape and scale "distances"
    '''
    threshold_shape = 1.
    threshold_scaling = 1.5
    thresholded_indices = []

    # load spike unit data
    spike_units = None
    if output is not None:

        makedirs_save(output)

        wf_file = op.join(output, 'spike_unit_data.npy')
        if op.exists(wf_file) and not overwrite:

            tmp = np.load(wf_file,
                          encoding='latin1',
                          allow_pickle=True)
            spike_units = [helpers.SpikeUnit(**dd) for dd in tmp]

    if spike_units is None:

        spike_units = load_spike_units(path)

    if output is not None:

        if not op.exists(wf_file) or overwrite:
            # workaround as save cannot handle class instance but object dict
            
            np.save(wf_file, [u.as_dict() for u in spike_units])

    # compute distances
    dist_shape_scale, unit_indices = compute_waveform_distances_between_units(spike_units)

    #CAP THE ARRAYS AT THE THRESHOLD
    for i, (d1, d2, index) in enumerate(zip(dist_shape_scale[:,0], dist_shape_scale[:,1], unit_indices)):

        if d1 <= threshold_shape and d2 <= threshold_scaling:

            thresholded_indices.append(index)

    thresholded_indices = np.asarray(thresholded_indices)

    if output is not None:

        days_between_recordings = get_days_between_recordings(path,
                                                              count_recordings=True)
        # save distances and unit information to file
        result_file = op.join(output, 'matched_units_waveforms.npz')
        print("saving result to", result_file)
        np.savez(result_file,
                 spike_units=[u.as_dict() for u in spike_units],
                 dist_shape_scale=dist_shape_scale,
                 unit_indices=unit_indices,
                 threshold_shape=threshold_shape,
                 threshold_scaling=threshold_scaling,
                 thresholded_indices=thresholded_indices,
                 days_between_recordings=days_between_recordings)

    # scatter plot of d_1 vs d_2
    fig, ax = plt.subplots()

    
    ax.scatter(dist_shape_scale[:, 0],
                dist_shape_scale[:, 1],
                s=4,
                c=3*[.5],
                edgecolors='none')
    ax.set_xlabel(r'$d_1$ (shape)')
    ax.set_ylabel(r'$d_2$ (scaling)')
    ax.axvline(threshold_shape, color='k', linewidth = 1.0)
    ax.axhline(threshold_scaling, color='k', linewidth = 1.0)

    scu.simple_xy_axes(ax)
    scu.set_font_axes(ax, add_size=8)

    fig.set_size_inches(10, 8)
    fig.tight_layout()
    if output is not None:
        fig.savefig(op.join(output, 'd1d2_plot.pdf'))
    

    if create_pdf and output is not None:
        # create pdf with units with distances below some user-defined threshold

        with PdfPages(op.join(output, 'matched_units_waveforms.pdf')) as pdf:

            for i, (d_shape, d_scaling) in enumerate(dist_shape_scale):

                if d_shape <= threshold_shape and d_scaling <= threshold_scaling:

                    i1, i2 = unit_indices[i]
                    u1 = spike_units[i1]
                    u2 = spike_units[i2]

                    fig, axes = plt.subplots(nrows=1, ncols=2,
                                             sharex=True,
                                             sharey=True)

                    shift = max(np.max(np.abs(u1.waveforms)),
                                np.max(np.abs(u2.waveforms)))
                    for j, u in enumerate([u1, u2]):
                        ax = axes[j]
                        ax.set_title('{}, tt {}, clu {} ({})'.format(
                            u.date.replace('_', '/'), u.tetrode, u.cluster, u.cluster_group))
                        helpers.plot_waveforms(ax, u.waveforms,
                                               channel_shift=shift)

                        scu.set_font_axes(ax)

                    fig.text(.08, .5, 'd_shape={:.2f}\nd_scaling={:.2f}'.format(d_shape, d_scaling),
                             ha='center',
                             va='center',
                             linespacing=1.5,
                             fontsize=8,
                             family='Arial')
                    fig.set_size_inches(5, 2.25)
                    fig.tight_layout(rect=(.1, 0, 1, .95),
                                     pad=.1,
                                     w_pad=.25)

                    pdf.savefig(fig)
                    plt.close(fig)

    plt.show(block=True)


cli.add_command(cli_tolias2007)



# make graph which summarises the number of cells detected across session
import os
import os.path as op

@click.command(name = 'summary')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(exists=False))
@click.option('--pre', '-p', default=.5)
@click.option('--post', '-P', default=.5)
@click.option('--binwidth', '-b', default=.025)
@click.option('--show', '-s', is_flag=True)
def cli_summary(db_path=None,
        output=None,
        units_per_page=4,
        show=False,
        **kwargs):

    days_recs = sorted(os.listdir(db_path))
    #for day in days_recs:
    sessions = [op.join(db_path, day) for day in days_recs if day.startswith('20')]
    session_paths = [op.join(path, os.listdir(path)[0]) for path in sessions]
    
    units_across = []

    for session_path in session_paths:
        print('analyzing:', session_path)
        units_and_grouping = []
        recordings = helpers.get_recordings(session_path)
        print(recordings)
        reader = NeoIO(session_path)
        block = reader.read_block(recordings=recordings,
                                  dtypes=['spikes'],
                                  cluster_groups=['Good', 'MUA'])
        units = block.list_units
        n_units = len(units)
        units_and_grouping.append(n_units)

        print("    segments:", len(block.segments))
        print("    units:", len(units))


        good = 0
        mua = 0
        if len(units) > 0:
            for unit in units:
                depth = unit.annotations['electrode_depth']
                if unit.annotations['group'] == 'Good':
                    good += 1
                elif unit.annotations['group'] == 'Mua':
                    mua += 1
            units_and_grouping.append(good)
            units_and_grouping.append(mua)
            units_and_grouping.append(depth)
            units_across.append(units_and_grouping)
            print(units_across)

    #PLOT IT ALL
#    import pdb
#    pdb.set_trace()
    units_across = np.vstack(units_across)
    fig, ax = plt.subplots()
    ax.plot(range(1,len(session_paths)), units_across[:,0],'k-', label= 'All')
    ax.plot(range(1,len(session_paths)), units_across[:,1],'g--', linewidth=0.25, label='Good')
    ax.plot(range(1,len(session_paths)), units_across[:,2],'b--', linewidth=0.25, label='Mua')
    ax.set_xlabel('Recording')
    ax.set_ylabel('N of units')
    ax2 = ax.twinx()
    ax2.plot(range(1,len(session_paths)), units_across[:,3],'m-', label= 'Depth')
    ax2.set_ylabel('Depth of electrodes (um)')
    fig.tight_layout()
    ax.legend()
    plt.show()



cli.add_command(cli_summary)


@click.command(name='average')
@click.argument('global_db_path', type=click.Path(exists=True))
def cli_average(global_db_path=None):

    days = []
    fig, (ax0, ax1) = plt.subplots(nrows=2)

    for animal in os.listdir(global_db_path):
        print('analysing:', animal)
        wavedata = np.load(op.join(global_db_path, animal,'matched_units_waveforms.npz'), encoding='latin1',
                                allow_pickle=True)
        for pair in wavedata['dist_shape_scale']:
            ax0.scatter(pair[0], pair[1], color='black', s=0.1)
        for day in wavedata['days_between_recordings']:
            days.append(day)
        ax0.axvline(wavedata['threshold_shape'], linestyle='--', linewidth=0.5, color='r')
        ax0.axhline(wavedata['threshold_scaling'], linestyle='--', linewidth=0.5, color='r')
    ax1.hist(days, color='grey', bins=max(days)-1, alpha=0.85, histtype='bar', ec='black', align='mid')
    ax1.set_xlim(1,30)
    ax1.set_xlabel('Days after first recording')
    ax1.set_ylabel('Number\nof matches')
    ax0.set_xlabel(r'$d_1$ (shape)')
    ax0.set_ylabel(r'$d_2$ (scaling)')
    #show or save figure
    plt.tight_layout()
    plt.show()
    for ff in ['png', 'pdf']:
        fig.savefig(op.join(global_db_path, 'upgrade_fig2_python.' + ff),
                    format=ff,
                    dpi=300)

cli.add_command(cli_average)

@click.command(name='null')
@click.argument('animal_path')
@click.option('--output', '-o', default=None)
@click.option('--nb_q_turn', '-n', default=2)
def cli_null(animal_path=None,
             output=None,
             nb_q_turn=2):
    
    # Calculate d1 and d2 between neurons of different depths for a null
    # distribution
    
    # user-defined thresholds for shape and scale "distances"
    threshold_shape = 1.
    threshold_scaling = 1.5
    thresholded_indices = []    
    
    spike_units = None
    spike_units = load_spike_units(animal_path)
    
    dist_shape_scale, unit_indices = compute_waveform_distances_null(
        spike_units, nb_q_turn)
    
    
    if output is not None:

        # save distances and unit information to file
        result_file = op.join(output, 'matched_units_waveforms_null_' + animal_path[:-1].split('/')[-2] + '.npz')
        print("saving result to", result_file)
        np.savez(result_file,
                 spike_units=[u.as_dict() for u in spike_units],
                 dist_shape_scale=dist_shape_scale,
                 unit_indices=unit_indices,
                 threshold_shape=threshold_shape,
                 threshold_scaling=threshold_scaling)
    
cli.add_command(cli_null)
            
    

@click.command(name='old_null')
@click.argument('global_db_path')
@click.option('--output', '-o', default=None)
def cli_old_null(global_db_path=None,
                   output=None):
    # Calculate d1 and d2 between different animal for a null distribution
    
    # user-defined thresholds for shape and scale "distances"
    threshold_shape = 1.
    threshold_scaling = 1.5
    thresholded_indices = []
    
    animals = os.listdir(global_db_path)
    animal_pairs = list(itertools.combinations(animals, 2))
    
    
    for pair in animal_pairs:
        animal1_path = op.join(global_db_path, pair[0], 'neural')
        animal2_path = op.join(global_db_path, pair[1], 'neural')
        spike_units1 = None
        spike_units2 = None
            
        spike_units1 = load_spike_units(animal1_path)
        spike_units2 = load_spike_units(animal2_path)
        
        # compute distances
        dist_shape_scale, unit_indices = compute_waveform_distances_old_null(
            spike_units1, spike_units2)
        
        #CAP THE ARRAYS AT THE THRESHOLD
        for i, (d1, d2, index) in enumerate(zip(dist_shape_scale[:,0], dist_shape_scale[:,1], unit_indices)):
    
            if d1 <= threshold_shape and d2 <= threshold_scaling:
    
                thresholded_indices.append(index)
    
        thresholded_indices = np.asarray(thresholded_indices)
        
        if output is not None:
        
            # save distances and unit information to file
            result_file = op.join(output, 'matched_units_waveforms_' + pair[0] + '_' + pair[1] + '.npz')
            print("saving result to", result_file)
            np.savez(result_file,
                     animal1 = pair[0],
                     animal2 = pair[1],
                     spike_units_1=[u.as_dict() for u in spike_units1],
                     spike_units_2=[u.as_dict() for u in spike_units2],
                     dist_shape_scale=dist_shape_scale,
                     unit_indices=unit_indices,
                     threshold_shape=threshold_shape,
                     threshold_scaling=threshold_scaling,
                     thresholded_indices=thresholded_indices)
                 
                 
cli.add_command(cli_old_null)



@click.command(name='EM_matching')
@click.argument('db_path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None)
@click.option('--nb_q_turn_null', '-n', default=4)
@click.option('--confidence', '-conf', default=0.99)
@click.option('--overwrite', '-O', is_flag=True)
@click.option('--create-pdf', '-c', is_flag=True)
def cli_EM_matching(db_path=None,
                output=None,
                nb_q_turn_null=4,
                confidence=0.99,
                overwrite=False,                
                create_pdf=False):

    '''This function calculate the null distribution from the null d1_d2 data
    using a Expectation-Maximization algorithm.
    
    Then, this null distribution is fixed to the experimental d1_d2 data to
    compute the matched distribution.
    
    The matched units are the ones that are only in the 99% intervall of the
    matched distribution'''
    
    wavedata = {}
    wavedata_null = {}
    
    # Compute d1_d2 as in tolias2007 for the experimental and the null
    # distribution for each animal
    list_animal = os.listdir(db_path)
    list_animal = [x for x in list_animal if x.startswith('M102') and not x.endswith('_neural')]
    for animal in list_animal:
        print('computing distances for', animal)
        path = op.join(db_path, animal)
        wavedata[animal] = {}
        wavedata_null[animal] = {}

        # load spike unit data
        spike_units = None
        if output is not None:
            
            animal_output = op.join(output, animal)
            makedirs_save(animal_output)
    
            wf_file = op.join(animal_output, 'spike_unit_data.npy')
            if op.exists(wf_file) and not overwrite:
    
                tmp = np.load(wf_file,
                              encoding='latin1',
                              allow_pickle=True)
                spike_units = [helpers.SpikeUnit(**dd) for dd in tmp]        
            
        if spike_units is None:
    
            spike_units = load_spike_units(path)
    
        if output is not None:
    
            if not op.exists(wf_file) or overwrite:
                # workaround as save cannot handle class instance but object dict
                
                np.save(wf_file, [u.as_dict() for u in spike_units])        
    
        # compute distances
        wavedata[animal]['dist_shape_scale'], wavedata[animal]['unit_indices'] , wavedata[animal]['days_between_recordings']\
            = compute_waveform_distances_between_units(spike_units)
        
        # wavedata[animal]['days_between_recordings'] = get_days_between_recordings(
        #                                                 path,
        #                                                 count_recordings=True)
        
        wavedata[animal]['spike_units'] = [u.as_dict() for u in spike_units]
        
        # compute null distances
        wavedata_null[animal]['dist_shape_scale'], wavedata_null[animal]['unit_indices']\
            = compute_waveform_distances_null(spike_units, nb_q_turn_null)
    #pdb.set_trace()
        
        
    # First the null distribution is computed from the null data
    dist_data_null = []
    for animal in wavedata_null:
        if len(wavedata_null[animal]['dist_shape_scale']) > 0:
               dist_data_null.append(wavedata_null[animal]['dist_shape_scale'][:][:])
    
    print('Computing null distribution')

    data_null = np.concatenate(dist_data_null)
    X_null = data_null[~np.isnan(data_null).any(axis=1)]
    
    # define the number of clusters to be learned
    k = 1
    
    # create and initialize the cluster centers and the weight paramters
    weights_null = np.ones((k)) / k
    means_null = np.random.choice(X_null.flatten(), (k,X_null.shape[1]))
    
    # create and initialize a Positive semidefinite convariance matrix 
    cov_null = []
    for i in range(k):
      cov_null.append(make_spd_matrix(X_null.shape[1]))
    cov_null = np.array(cov_null)    
    
    eps=1e-8
    # run GMM for 40 steps
    for step in range(40):
        
        likelihood = []
        # Expectation step
        for j in range(k):
          likelihood.append(multivariate_normal.pdf(x=X_null, mean=means_null[j], cov=cov_null[j]))
        likelihood = np.array(likelihood)
        assert likelihood.shape == (k, len(X_null))
          
        b = []
        # Maximization step 
        for j in range(k):
          # use the current values for the parameters to evaluate the posterior
          # probabilities of the data to have been generanted by each gaussian
          b.append((likelihood[j] * weights_null[j]) / (np.sum([likelihood[i] * weights_null[i] for i in range(k)], axis=0)+eps))
          
          # updage mean and variance
          means_null[j] = np.sum(b[j].reshape(len(X_null),1) * X_null, axis=0) / (np.sum(b[j]+eps))
          cov_null[j] = np.dot((b[j].reshape(len(X_null),1) * (X_null - means_null[j])).T, (X_null - means_null[j])) / (np.sum(b[j])+eps)
          
          # update the weights
          weights_null[j] = np.mean(b[j])
          
          assert cov_null.shape == (k, X_null.shape[1], X_null.shape[1])
          assert means_null.shape == (k, X_null.shape[1])        
    
    
    # Then the distribution of matches is computed from the experimental data
    # using the null distribution previously obtained
    
    dist_data = []
    for animal in wavedata:
        dist_data.append(wavedata[animal]['dist_shape_scale'][:][:])
    
    print('Computing experimental distribution')

    data = np.concatenate(dist_data)
    X = data[~np.isnan(data).any(axis=1)]
    
    # define the number of clusters to be learned
    k = 2
    
    # create and initialize the cluster centers and the weight paramters
    weights = np.ones((k)) / k
    means = np.random.choice(X.flatten(), (1,X.shape[1]))
    means = np.array([means[0], means_null[0]])
    
    # create and initialize a Positive semidefinite convariance matrix 
    cov = []
    for i in range(1):
      cov.append(make_spd_matrix(X.shape[1]))
    cov = np.array(cov)
    cov = np.array([cov[0], cov_null[0]])
    
    eps=1e-8
    # run GMM for 40 steps while fixing the null distribution
    for step in range(40):    
    
        likelihood = []
        # Expectation step
        for j in range(k):
          likelihood.append(multivariate_normal.pdf(x=X, mean=means[j], cov=cov[j]))
        likelihood = np.array(likelihood)
        assert likelihood.shape == (k, len(X))
          
        b = []
        # Maximization step 
        for j in range(1):
          # use the current values for the parameters to evaluate the posterior
          # probabilities of the data to have been generated by each gaussian
          b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))
          
          # updage mean and variance
          means[j] = np.sum(b[j].reshape(len(X),1) * X, axis=0) / (np.sum(b[j]+eps))
          cov[j] = np.dot((b[j].reshape(len(X),1) * (X - means[j])).T, (X - means[j])) / (np.sum(b[j])+eps)
          
          # update the weights
          weights[j] = np.mean(b[j])
          
          assert cov.shape == (k, X.shape[1], X.shape[1])
          assert means.shape == (k, X.shape[1])    
         
            
    
            
    r_ellipse_match, object_ellipse_match = get_ellipse(means[0], cov[0],
                                                        confidence=confidence)
    r_ellipse_null, object_ellipse_null = get_ellipse(means[1], cov[1],
                                                      confidence=confidence)
    
    if create_pdf and output is not None:
        
        # Use pandas to analyse the data
        X_null = pd.DataFrame(data = {r'$d_1$ (shape)': X_null[:,0], r'$d_2$ (scaling)': X_null[:,1]})
        X = pd.DataFrame(data = {r'$d_1$ (shape)': X[:,0], r'$d_2$ (scaling)': X[:,1]})
        
        gnull = sns.JointGrid(x=r'$d_1$ (shape)', y=r'$d_2$ (scaling)', data=X_null)
        gnull = gnull.plot_joint(sns.scatterplot, s=8, color=scu.NICE_COLORS['black'])
        ax = gnull.ax_joint
        ax.plot(r_ellipse_null[:,0],r_ellipse_null[:,1],'--', c = scu.NICE_COLORS['blue'], label='Null distribution 95% CI')
        
        ax.set_xlim([-1, 8])
        ax.set_ylim([-0.5,25])
        
        gnull = gnull.plot_marginals(sns.kdeplot, bw=.2, color=scu.NICE_COLORS['gray'], shade=True)        
        
        
        gexp = sns.JointGrid(x=r'$d_1$ (shape)', y=r'$d_2$ (scaling)', data=X)
        gexp = gexp.plot_joint(sns.scatterplot, s=8, color=scu.NICE_COLORS['black'])
        ax = gexp.ax_joint
        ax.plot(r_ellipse_null[:,0],r_ellipse_null[:,1],'--', c = scu.NICE_COLORS['blue'], label='Null distribution 95% CI')
        ax.plot(r_ellipse_match[:,0],r_ellipse_match[:,1],'--', c = scu.NICE_COLORS['dark orange'], label='Experimental distribution 95% CI')
              
        ax.set_xlim([-1, 8])
        ax.set_ylim([-0.5,25])
        
        gexp = gexp.plot_marginals(sns.kdeplot, bw=.2, color=scu.NICE_COLORS['gray'], shade=True)
        #gexp = gexp.plot_marginals(sns.distplot, color=scu.NICE_COLORS['gray'])
                
        for g in [gnull, gexp]:
            for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
                scu.set_font_axes(ax, add_size=15)
                
                
        fig = plt.figure(figsize=(16,8))
        gs = gridspec.GridSpec(1,2)
        
        mg0 = helpers.SeabornFig2Grid(gnull, fig, gs[0])
        mg1 = helpers.SeabornFig2Grid(gexp, fig, gs[1])
        
        gs.tight_layout(fig)

        fig.savefig(op.join(output, 'distributions_matched.pdf'),
                    bbox_inches='tight')
        
        
    
    for animal in wavedata:
        print('Determining matches for:', animal)
        thresholded_indices = []
        dist_shape_scale = wavedata[animal]['dist_shape_scale']
        unit_indices = wavedata[animal]['unit_indices']
        
        # Select pairs within the 95% CI of the matched distribution 
        # but not the null distribution
        for i, (dist, index) in enumerate(zip(dist_shape_scale, unit_indices)):
            
            if object_ellipse_match.contains_point(dist) & ~ object_ellipse_null.contains_point(dist):

                thresholded_indices.append(index)
                
        # /!\ manually remove some matches for M10234 where some units were
        # matched with units from the same day /!\
              
        if animal == 'M10234':
            thresholded_indices = remove_multiple_matched_day(unit_indices,
                                        thresholded_indices,
                                        dist_shape_scale,
                                        r_ellipse_null)
            
                
        if output is not None:
            
            animal_output = op.join(output, animal)
            
            result_file = op.join(animal_output, 'matched_units_waveforms.npz')
            np.savez(result_file,
                 spike_units=wavedata[animal]['spike_units'],
                 dist_shape_scale=wavedata[animal]['dist_shape_scale'],
                 unit_indices=wavedata[animal]['unit_indices'],
                 thresholded_indices=thresholded_indices,
                 days_between_recordings=wavedata[animal]['days_between_recordings'])

        
        if create_pdf and output is not None:
            # create pdf with units with distances below some user-defined threshold
    
            spike_units = wavedata[animal]['spike_units']
            with PdfPages(op.join(animal_output, 'matched_units_waveforms.pdf')) as pdf:
    
                for i, (d_shape, d_scaling) in enumerate(dist_shape_scale):
            
                    if object_ellipse_match.contains_point([d_shape, d_scaling]) & ~ object_ellipse_null.contains_point([d_shape, d_scaling]):

    
                        i1, i2 = unit_indices[i]
                        u1 = spike_units[i1]
                        u2 = spike_units[i2]
    
                        fig, axes = plt.subplots(nrows=1, ncols=2,
                                                 sharex=True,
                                                 sharey=True)
    
                        shift = max(np.max(np.abs(u1['waveforms'])),
                                    np.max(np.abs(u2['waveforms'])))
                        for j, u in enumerate([u1, u2]):
                            ax = axes[j]
                            ax.set_title('{}, tt {}, clu {} ({})'.format(
                                u['date'].replace('_', '/'), u['tetrode'], u['cluster'], u['cluster_group']))
                            helpers.plot_waveforms(ax, u['waveforms'],
                                                   channel_shift=shift)
    
                            scu.set_font_axes(ax)
    
                        fig.text(.08, .5, 'd_shape={:.2f}\nd_scaling={:.2f}'.format(d_shape, d_scaling),
                                 ha='center',
                                 va='center',
                                 linespacing=1.5,
                                 fontsize=8,
                                 family='Arial')
                        fig.set_size_inches(5, 2.25)
                        fig.tight_layout(rect=(.1, 0, 1, .95),
                                         pad=.1,
                                         w_pad=.25)
    
                        pdf.savefig(fig)
                        plt.close(fig)        
    
cli.add_command(cli_EM_matching)


#------------- Some plotting functions for esveform matching figure


def plot_waveforms_examples(fig, ax, waveform_matching_path):
    # waveform_matching_path = '/Volumes/groupfolders/DBIO_LindenLab_DATA/DATA_ANAT/jules/analysis/EM_waveform_matching/250_microns/99_CI/M10234/'
    #waveform_matching_path = '/Volumes/LACIE_SHARE/LindenLab/analysis/waveform_matching/M10234/'
    matched_data = np.load(op.join(waveform_matching_path, 'matched_units_waveforms.npz'), allow_pickle=True, encoding="latin1")
    # here waveformmatching_path = path for m10234 because only examples from this animl are used in the figure.
    spike_units = matched_data['spike_units']
    thresholded_indices = matched_data['thresholded_indices']
    unit_indices = matched_data['unit_indices']
    dist_shape_scale = matched_data['dist_shape_scale']

    #i1, i2 = thresholded_indices[2] # 95CI example
    i1, i2 = thresholded_indices[26] # 99CI example
    u1 = spike_units[i1]
    u2 = spike_units[i2]
    #create a grid within the subplot to plot the five examples
    axes = gridspec.GridSpecFromSubplotSpec(6, 4,
                    subplot_spec=ax, wspace=0.1, hspace=0.3)
    #play around with wspace and hspace to make the shape of the waveforms different

    axx = [fig.add_subplot(axes[:2,0]), fig.add_subplot(axes[:2,1])] # for the first and second wavefrom of thesame comparison
    shift = max(np.max(np.abs(u1['waveforms'])),
                np.max(np.abs(u2['waveforms'])))
    for j, u in enumerate([u1, u2]):
        ax = axx[j]
        helpers.plot_waveforms(ax, u['waveforms'],
                               channel_shift=shift,
                              color='lime') #change this to have changed colors

        scu.set_font_axes(ax)


    i1, i2 = thresholded_indices[8] #use the numbers of pages on the pdf of the comparisons to change this index if you want a different example
    u1 = spike_units[i1]
    u2 = spike_units[i2]

    axx = [fig.add_subplot(axes[2:4,0]), fig.add_subplot(axes[2:4,1])]
    shift = max(np.max(np.abs(u1['waveforms'])),
                np.max(np.abs(u2['waveforms'])))
    for j, u in enumerate([u1, u2]):
        ax = axx[j]
        helpers.plot_waveforms(ax, u['waveforms'],
                               channel_shift=shift,
                              color='darkgreen')

        scu.set_font_axes(ax)

    i1, i2 = thresholded_indices[3]
    u1 = spike_units[i1]
    u2 = spike_units[i2]

    axx = [fig.add_subplot(axes[4:,0]), fig.add_subplot(axes[4:,1])]
    shift = max(np.max(np.abs(u1['waveforms'])),
                np.max(np.abs(u2['waveforms'])))
    for j, u in enumerate([u1, u2]):
        ax = axx[j]
        helpers.plot_waveforms(ax, u['waveforms'],
                               channel_shift=shift,
                              color='teal')

        scu.set_font_axes(ax)    

    i1, i2 = unit_indices[201]
    u1 = spike_units[i1]
    u2 = spike_units[i2]
    
    axx = [fig.add_subplot(axes[1:3,2]), fig.add_subplot(axes[1:3,3])]
    shift = max(np.max(np.abs(u1['waveforms'])),
                np.max(np.abs(u2['waveforms'])))
    for j, u in enumerate([u1, u2]):
        ax = axx[j]
        helpers.plot_waveforms(ax, u['waveforms'],
                               channel_shift=shift,
                              color=scu.NICE_COLORS['violet'])

        scu.set_font_axes(ax)


    i1, i2 = unit_indices[200]
    u1 = spike_units[i1]
    u2 = spike_units[i2]
    
    #axx = axes[4]
    axx = [fig.add_subplot(axes[3:5,2]), fig.add_subplot(axes[3:5,3])]

    
    shift = max(np.max(np.abs(u1['waveforms'])),
                np.max(np.abs(u2['waveforms'])))
    for j, u in enumerate([u1, u2]):
        ax = axx[j]
        helpers.plot_waveforms(ax, u['waveforms'],
                               channel_shift=shift,
                              color=scu.NICE_COLORS['orchid'])

        scu.set_font_axes(ax)


def plot_distribs(X_null, X, r_ellipse_null, r_ellipse_match,
                  dist_shape_scale,
                  unit_indices,
                  thresholded_indices):
    #create panel A
    gnull = sns.JointGrid(x=r'$d_1$ (shape)', y=r'$d_2$ (scaling)', data=X_null)
    gnull = gnull.plot_joint(sns.scatterplot, s=8, color=scu.NICE_COLORS['black'])
    ax = gnull.ax_joint
    ax.plot(r_ellipse_null[:,0],r_ellipse_null[:,1],'--', c = scu.NICE_COLORS['blue'], label='Null distribution 99% CI')

    ax.set_xlim([-1, 8])
    ax.set_ylim([-0.5,25])

    gnull = gnull.plot_marginals(sns.kdeplot, bw=.2, color=scu.NICE_COLORS['gray'], shade=True) #the marginals come now

    #sme thing for panel B
    gexp = sns.JointGrid(x=r'$d_1$ (shape)', y=r'$d_2$ (scaling)', data=X)
    gexp = gexp.plot_joint(sns.scatterplot, s=8, color=scu.NICE_COLORS['black'])
    ax = gexp.ax_joint

    ax.plot(r_ellipse_null[:,0],r_ellipse_null[:,1],'--', c = scu.NICE_COLORS['blue'], label='Null distribution 99% CI')
    ax.plot(r_ellipse_match[:,0],r_ellipse_match[:,1],'--', c = scu.NICE_COLORS['red'], label='Experimental distribution 99% CI')

    #plot colored dots: make sure indices and colours are the same as in plot_waveforms_examples
    #np.wheree finds thee index of unit indices where your thresholded unit is
    #counter makes a dictionary with the times each one occurs
    #we need the one which appears twice (i.e.: is a match)
    cnt = Counter(np.where(unit_indices == thresholded_indices[26])[0])
    #this line identifies the index of the match in interest
    dist_idx = [k for k, v in cnt.items() if v > 1][0]
    ax.scatter(dist_shape_scale[dist_idx][0], dist_shape_scale[dist_idx][1], 40, color= 'lime')

    cnt = Counter(np.where(unit_indices == thresholded_indices[8])[0])
    dist_idx = [k for k, v in cnt.items() if v > 1][0]
    ax.scatter(dist_shape_scale[dist_idx][0], dist_shape_scale[dist_idx][1], 40, color='darkgreen')

    cnt = Counter(np.where(unit_indices == thresholded_indices[3])[0])
    dist_idx = [k for k, v in cnt.items() if v > 1][0]
    ax.scatter(dist_shape_scale[dist_idx][0], dist_shape_scale[dist_idx][1], 40, color= 'teal')

    #these two are not matches and so we just find the index from pdf and plot

    ax.scatter(dist_shape_scale[201][0], dist_shape_scale[201][1], 40, color=scu.NICE_COLORS['violet'] )

    ax.scatter(dist_shape_scale[200][0], dist_shape_scale[200][1], 40, color=scu.NICE_COLORS['orchid'] )



    ax.set_xlim([-1, 8])
    ax.set_ylim([-0.5,25])

    gexp = gexp.plot_marginals(sns.kdeplot, bw=.2, color=scu.NICE_COLORS['gray'], shade=True)

    return(gnull, gexp)


def plot_matches_days(ax, matched_db):
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)

    # matched_db = '/Volumes/groupfolders/DBIO_LindenLab_DATA/DATA_ANAT/jules/analysis/EM_waveform_matching/250_microns/99_CI/'

    matched_neurons = {}
    list_animal = os.listdir(matched_db)
    list_animal = [x for x in list_animal if x.startswith('M102') and not x.endswith('_neural')]
    for animal in list_animal:
        matched_neurons[animal] = np.load(op.join(matched_db, animal, 'matched_units_waveforms.npz'), allow_pickle=True, encoding="latin1")
    
    days_max = max(matched_neurons['M10234']['days_between_recordings'])
    n_matched_cells = np.zeros(days_max)
    n_all_cells = np.zeros(days_max)
    for animal in list_animal:
        #get matches
        for [i1, i2] in matched_neurons[animal]['thresholded_indices']:
            u1 = matched_neurons[animal]['spike_units'][i1]
            u2 = matched_neurons[animal]['spike_units'][i2]
            days = get_days_between_units(u1, u2)
            n_matched_cells[days - 1]+=1
        #get all comparisons that were made
        for [i1, i2] in matched_neurons[animal]['unit_indices']:
            u1 = matched_neurons[animal]['spike_units'][i1]
            u2 = matched_neurons[animal]['spike_units'][i2]
            days = get_days_between_units(u1, u2)
            n_all_cells[days - 1]+=1

    percentage_matched = (n_matched_cells/n_all_cells)*100
    days_diff = np.arange(1, days_max + 1)
    ax.bar(days_diff, n_matched_cells, color = 'k')
    ax.set_xlabel('Days between recordings')
    ax.set_ylabel('Matches')
    ax2 = ax.twinx()
    ax2.plot(days_diff, percentage_matched, color = 'grey', linestyle = '--')
    ax2.set_ylabel('Percentage Matched', color = 'grey')
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    helpers.adjust_spines(ax, ['left', 'bottom'])
    top_side = ax2.spines["top"]
    top_side.set_visible(False)
    bottom_side = ax2.spines["bottom"]
    bottom_side.set_visible(False)
    left_side = ax2.spines["left"]
    left_side.set_visible(False)
    helpers.adjust_spines(ax2, ['right' ])
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
    scu.set_font_axes(ax2)
    # data_matched = pd.DataFrame(data = {'Days between recordings' : days_diff,
    #                                     'Matches' : n_matched_cells})
    # sns.set()
    # sns.set_style("ticks")
    # sns.set_context('paper')
    # ax = sns.barplot(x='Days between recordings', y='Matches',
    #                  data=data_matched,
    #                 color=scu.NICE_COLORS['gray'], ax=ax)
    # sns.despine(offset=5, ax=ax)
    # ax.yaxis.set_major_locator(MultipleLocator(50))
    # ax.xaxis.set_major_locator(MultipleLocator(3))

    #if the remainder of the division n/5 is not 0 then remove the label
    # every_nth_x = 5
    # for n, label in enumerate(ax.xaxis.get_major_ticks()):
    #     if n % every_nth_x != 0:
    #         label.set_visible(False)


if __name__ == '__main__':
    cli()
