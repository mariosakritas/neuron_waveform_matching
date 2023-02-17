#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting functions for databse analysis prior to wf matching

@author: marios
"""


from __future__ import print_function
import click
import numpy as np
import matplotlib.pyplot as plt
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



@click.group()
def cli():
    pass

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
# make graph which summarises the number of cells detected across session
# -----------------------------------------------------------------------------

@click.command(name='summary')
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
    # for day in days_recs:
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

    # PLOT IT ALL
    #    import pdb
    #    pdb.set_trace()
    units_across = np.vstack(units_across)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(session_paths)), units_across[:, 0], 'k-', label='All')
    ax.plot(range(1, len(session_paths)), units_across[:, 1], 'g--', linewidth=0.25, label='Good')
    ax.plot(range(1, len(session_paths)), units_across[:, 2], 'b--', linewidth=0.25, label='Mua')
    ax.set_xlabel('Recording')
    ax.set_ylabel('N of units')
    ax2 = ax.twinx()
    ax2.plot(range(1, len(session_paths)), units_across[:, 3], 'm-', label='Depth')
    ax2.set_ylabel('Depth of electrodes (um)')
    fig.tight_layout()
    ax.legend()
    plt.show()


cli.add_command(cli_summary)


if __name__ == '__main__':
    cli()
