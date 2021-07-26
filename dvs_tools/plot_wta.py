"""
Copyright © 2018 University of Zurich.

"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms
from pyqtgraph.Qt import QtGui

from loihi_tools.plotter2d import Plotter2d, cm_on_off


def ident(x):
    return x


def plot_wta(file='./results/loihi_spikes.npz', dims_exc=(120, 90),
             dt_ms=0.5, filtersize_ms=10, frames=None, frames_timestamps=None,
             num_rows=1, num_panes=10, generate_gif=True, show_plot=False):
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))

    lm = {}
    with np.load(file, allow_pickle=True) as loaded:
        for arr in loaded:
            print(arr)
            lm[arr] = loaded[arr].item()

    dt = dt_ms * ms
    filtersize_ms = filtersize_ms * ms
    plot_dt = 10 * ms

    win = QtGui.QDialog()
    gridlayout = QtGui.QGridLayout(win)

    counter = 0
    for ii, key in enumerate(lm):
        if key not in ['glinh', ]:
            filename = file.replace('.npz', '') + '_' + str(key)
            lm[key].t = lm[key].t * dt
            spikes_plotter = Plotter2d(lm[key], dims=dims_exc)

            if key == 'inp':
                if frames is not None:
                    spikes_plotter._frames = frames
                    spikes_plotter._frame_timestamps = frames_timestamps
                    imv = spikes_plotter.plot_frames()
                    gridlayout.addWidget(imv, 1, counter)
                    counter += 1
                    # imv.show()
                    # app.exec()
                    if generate_gif:
                        spikes_plotter.generate_movie(filename + 'frames.gif', scale='256x256', speed=1,
                                                      plotfunction='plot_frames',
                                                      plot_dt=44 * ms, filtersize=44 * ms)
                    spikes_plotter.plot_panes(num_panes=num_panes, timestep=None, filtersize=440 * ms,
                                              num_rows=num_rows,
                                              plotfunction='plot_frames', filename=filename + '_frames',
                                              colormap=ident)

                imv = spikes_plotter.plot3d_on_off(plot_dt=5 * ms, filtersize=10 * ms)
                gridlayout.addWidget(imv, 1, counter)
                counter += 1
                if generate_gif:
                    spikes_plotter.generate_movie(filename + '.gif', scale='256x256', speed=1,
                                                  plotfunction='plot3d_on_off',
                                                  plot_dt=5 * ms, filtersize=10 * ms)
                spikes_plotter.plot_panes(num_panes=num_panes, timestep=None, filtersize=10 * ms, num_rows=num_rows,
                                          plotfunction='plot3d_on_off', filename=filename + '_on_off',
                                          colormap=cm_on_off)
                spikes_plotter.plot_panes(num_panes=num_panes, timestep=None, filtersize=filtersize_ms,
                                          num_rows=num_rows,
                                          filename=filename + '_rate')
            else:
                imv = spikes_plotter.plot3d(plot_dt=plot_dt, filtersize=filtersize_ms)
                gridlayout.addWidget(imv, 1, counter)
                counter += 1
                # imv.show()
                # app.exec()
                if generate_gif:
                    spikes_plotter.generate_movie(filename + '.gif', scale='256x256', speed=1, plotfunction='plot3d',
                                                  plot_dt=plot_dt, filtersize=filtersize_ms)
                spikes_plotter.plot_panes(num_panes=num_panes, timestep=None, filtersize=filtersize_ms,
                                          num_rows=num_rows,
                                          filename=filename + '_rate')

    win.resize(counter * 400, 600)
    win.setLayout(gridlayout)
    win.setWindowTitle('DVS tracker')
    win.show()
    if show_plot:
        app.exec()

    try:
        plt.figure()
        plt.plot(lm['glinh'].t * dt, lm['glinh'].i, '.')
        plt.show()
    except KeyError:
        pass
    win.close()
    # app.aboutToQuit.connect()
    # app.quit()
    # sys.exit()

    return spikes_plotter

def plot_trajectory(traj):
    traj.shape[0]
    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1:8])
    plt.legend(['1', '2', '3'])
