#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit (waveform) tracking across sessions using and expanding the
algorithm described in

Tolias, A. S.; Ecker, A. S.; Siapas, A. G.; Hoenselaar, A.; Keliris, G. A. & Logothetis, N. K.
Recording chronically from the same neurons in awake, behaving primates.
Journal of neurophysiology, 2007, 98, 3780-3790

@author: arne, jules, marios
"""


from __future__ import print_function

import os.path as op
import os
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
import os
import os.path as op

from stimcontrol.io.database import NeoIO
from stimcontrol.util import makedirs_save
import stimcontrol.util as scu



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




@click.group()
def cli():
    pass

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

    '''This function calculates the null distribution from the null d1_d2 data
    using a Expectation-Maximization algorithm.
    
    Then, this null distribution is fixed to the experimental d1_d2 data to
    compute the matched distribution.
    
    The matched units are the ones that are only in the 99% intervall of the
    matched distribution and outside the 99& interval of the null.'''
    
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
        wavedata[animal]['spike_units'] = [u.as_dict() for u in spike_units]
        # compute null distances
        wavedata_null[animal]['dist_shape_scale'], wavedata_null[animal]['unit_indices']\
            = compute_waveform_distances_null(spike_units, nb_q_turn_null)

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



if __name__ == '__main__':
    cli()
