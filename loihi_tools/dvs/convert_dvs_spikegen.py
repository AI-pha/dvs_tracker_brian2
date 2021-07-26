import csv
import sys

import numpy as np

from loihi_tools.plotter2d import DVSmonitor, Plotter2d


def convert_dvs_to_spikegen_npz(file, target_dims=None, source_dims=(240, 180), dt_ms=1.0, plotrange=None, plot=False):
    """
    This function reads an aedat recording, plots it and saves an npz file that can be used to create a spikegenerator

    :param file:
    :param target_dims:
    :param source_dims:
    :param dt:
    :param plotrange:
    :param plot:
    :return:
    """

    from brian2 import ms, us
    dt = dt_ms * ms

    if file.endswith('.txt') or file.endswith('.csv'):
        with open(file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='	')
            dvsnp = np.asarray([row for row in csvreader]).astype(int).T
            if dvsnp.shape[0] == 3:
                dvsnp = np.concatenate((dvsnp, np.zeros((1, dvsnp.shape[1]))))
            dvs_mon = DVSmonitor(*dvsnp, unit=us)
    elif file.endswith('.aedat'):
        from teili import aedat2numpy
        # If you want this to work, you need to: pip install teili
        dvsnp = aedat2numpy(file, length=0, version='V2', debug=0, camera='DAVIS240', unit='ms')
        # dvs_mon = DVSmonitor(*dvsnp)

        dvs_mon = DVSmonitor(*dvsnp)

    if target_dims is not None:
        dvs_mon = spatial_downsampling(dvs_mon, target_dims=target_dims, source_dims=source_dims)
    else:
        target_dims = source_dims

    remove_time_offset = True
    if remove_time_offset:
        dvs_mon.t = dvs_mon.t - np.min(dvs_mon.t)

    dvs_plotter = Plotter2d(dvs_mon, dims=target_dims, plotrange=plotrange)

    if plot:
        from pyqtgraph.Qt import QtGui
        app = QtGui.QApplication.instance()
        if app is None:
            app = QtGui.QApplication(sys.argv)
        else:
            print('QApplication instance already exists: %s' % str(app))

        imv = dvs_plotter.plot3d_on_off(plot_dt=dt, filtersize=5 * ms)
        imv.show()
        app.exec()

        imv = dvs_plotter.plot3d(plot_dt=dt, filtersize=5 * ms)
        imv.show()
        app.exec()

    spiketimes = np.asarray(dvs_plotter.t / dt, dtype=int)
    spiketimes = spiketimes - np.min(spiketimes)

    numPorts = np.prod(target_dims)

    outfilename = file + '_' + str(target_dims[0]) + 'x' + str(target_dims[1]) + '_nx_spikegen.npz'
    np.savez(outfilename,
             numPorts=numPorts,
             indices=dvs_plotter.i,
             spiketimes=spiketimes
             )

    dvs_plotter.generate_movie(outfilename.replace('.npz', '_rate.gif'), scale='256x256', speed=1,
                               plotfunction='plot3d',
                               plot_dt=5 * ms, filtersize=10 * ms)
    dvs_plotter.generate_movie(outfilename.replace('.npz', '_events.gif'), scale='256x256', speed=1,
                               plotfunction='plot3d_on_off',
                               plot_dt=5 * ms, filtersize=10 * ms)

    print('file for spikegen saved as', outfilename)


def spatial_downsampling(dvs_mon, source_dims, downsampling_factor=None, target_dims=None, verbose=True):
    """
        This downsamples the input x,y coordinates by applyng the spatial kernel
        defined in downsampling_kernel.
    """
    if downsampling_factor is not None:
        n_x = source_dims[0] // downsampling_factor[0]
        n_y = source_dims[1] // downsampling_factor[1]
        if target_dims is not None:
            raise Exception('target_dims and downsampling_factor are both specified, only specify one!')
    else:
        if target_dims is not None:
            downsampling_factor = [0, 0]
            downsampling_factor[0] = source_dims[0] / target_dims[0]
            downsampling_factor[1] = source_dims[1] / target_dims[1]
            n_x = target_dims[0]
            n_y = target_dims[1]
        else:
            raise Exception('please specify either target_dims or downsampling_factor!')

    x = (dvs_mon.xi / downsampling_factor[0]).astype(int)
    y = (dvs_mon.yi / downsampling_factor[1]).astype(int)

    indices_to_keep = np.where((x < n_x) & (y < n_y))[0]

    dvs_mon_ds = DVSmonitor(x[indices_to_keep], y[indices_to_keep], dvs_mon.t[indices_to_keep],
                            dvs_mon.pol[indices_to_keep])

    if verbose:
        print('Downsampling camera input size by factor {d1}x{d2}:'.format(d1=downsampling_factor[0],
                                                                           d2=downsampling_factor[1]))
        if (source_dims[0] % downsampling_factor[0]):
            cut_col = int(source_dims[0] % downsampling_factor[0])
            print('Cutting the {:d} rightmost columns from the input stimulus'.format(cut_col))

        if (source_dims[1] % downsampling_factor[1]):
            cut_row = int(source_dims[1] % downsampling_factor[1])
            print('Cutting the last {:d} rows from the input stimulus'.format(cut_row))
        print('number of precessed events:', np.size(dvs_mon.xi))
        print('downsampled shape is', n_x, 'x', n_y)

    return dvs_mon_ds


def add_padding(dvs_mon, pad):
    '''
    adds pad to each index
    :return:
    '''
    dvs_mon.xi = dvs_mon.xi + pad
    dvs_mon.yi = dvs_mon.yi + pad
    return dvs_mon


def reshape(dvs_mon, bot_left_corner, top_right_corner):
    dx = top_right_corner[0] - bot_left_corner[0]
    dy = top_right_corner[1] - bot_left_corner[1]
    assert dx >= 0, "The top right corner is not to the right of the bottom left corner."
    assert dy >= 0, "The top right corner is not higher than the bottom left corner."

    x = dvs_mon.xi - (bot_left_corner[0] - 1)
    y = dvs_mon.yi - (bot_left_corner[1] - 1)

    indices_to_keep = np.where((x > 0) & (y > 0) & (x < (top_right_corner[0] - 1)) & (y < (top_right_corner[1] - 1)))[0]
    reshaped_dvs = DVSmonitor(x[indices_to_keep], y[indices_to_keep], dvs_mon.t[indices_to_keep],
                              dvs_mon.pol[indices_to_keep])

    source_dims = (top_right_corner[0] - bot_left_corner[0] + 1, top_right_corner[1] - bot_left_corner[1] + 1)
    return reshaped_dvs, source_dims


if __name__ == '__main__':
    from brian2 import ms, us, second

    file = './data/DAVIS240C-2018-10-12_2.aedat'
    convert_dvs_to_spikegen_npz(file, target_dims=(64, 64), source_dims=(240, 180), dt_ms=0.5,
                                plotrange=(0 * second, 4 * second), plot=True)
