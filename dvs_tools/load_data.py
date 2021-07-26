import os
import sys

import imageio
import numpy as np

from loihi_tools.dvs.convert_dvs_spikegen import spatial_downsampling, add_padding
from loihi_tools.plotter2d import DVSmonitor, Plotter2d


def load_frames(datapath):
    # load all images from images.txt and create a 3d numpy array
    inputfile = os.path.join(datapath, 'images.txt')
    timestamps, files = zip(*np.genfromtxt(inputfile, dtype=None, encoding='utf-8'))

    sample_im = imageio.imread(os.path.join(datapath, files[0]))
    frames = np.zeros((len(files), sample_im.shape[0], sample_im.shape[1]))
    for i, file in enumerate(files):
        frames[i] = imageio.imread(os.path.join(datapath, file))
    return frames, np.asarray(timestamps)


def convert_txt2npz(datapath, filename, inputfile=None, num_rows=None):
    if inputfile is None:
        inputfile = os.path.join(datapath, filename)
    event_data = np.genfromtxt(inputfile, max_rows=num_rows)
    np.savez(inputfile + '.npz', event_data=event_data)


def load_data(plot=False, downsampling_factor=None, target_dims=None, inputfile=None,
              num_rows=3000000, max_time=None, padding=0):
    """

    :param plot:
    :param downsampling_factor:
    :param target_dims:
    :param inputfile:
    :param num_rows:
    :param padding: padding adds empty space around, it will scale down to fit in target_dims
    :return:
    """

    os.path.isfile(inputfile)

    if inputfile.endswith('.aedat'):
        from teili import aedat2numpy
        events = aedat2numpy(inputfile, length=0, version='V2', debug=0, camera='DAVIS240', unit='ms')
        events[2, :] = events[2, :] - np.min(events[2, :])
    elif inputfile.endswith('.hdf5'):
        import h5py
        f = h5py.File(inputfile, 'r')
        event_data = np.asarray(f['davis_mono']['events'])

        events = np.zeros_like(event_data.T, dtype=int)
        events[0, :] = np.asarray(event_data[:, 0], dtype=int)
        events[1, :] = np.asarray(event_data[:, 1], dtype=int)
        events[2, :] = np.asarray(event_data[:, 2] * 10 ** 3, dtype=int)
        events[3, :] = np.asarray(event_data[:, 3], dtype=int)

        events[3, :][events[3, :] == 1] = 0
        events[3, :][events[3, :] == -1] = 1

        events[2, :] = events[2, :] - min(events[2, :])

    else:
        event_data = np.load(inputfile)['event_data']
        # event_data = np.genfromtxt(inputfile, max_rows=num_rows)
        # data is:
        # timestamp x y pol
        # change it to format aedat2numpy uses:
        events = np.zeros_like(event_data.T, dtype=int)
        events[0, :] = np.asarray(event_data[:, 1], dtype=int)  # x
        events[1, :] = np.asarray(event_data[:, 2], dtype=int)  # y
        events[2, :] = np.asarray(event_data[:, 0] * 10 ** 3, dtype=int)  # time
        events[3, :] = np.asarray(event_data[:, 3], dtype=int)  # polarity

    if max_time is not None:
        max_ind = np.where(events[2, :] < max_time)[0][-1]
        if max_ind < num_rows:
            events = events[:, 0:max_ind]
        else:
            events = events[:, 0:num_rows]
    else:
        events = events[:, 0:num_rows]
    dvs_mon = DVSmonitor(*events)
    # np.max(dvs_mon.t)
    source_dims = (240, 180)
    if downsampling_factor is not None:
        n_x = source_dims[0] // downsampling_factor[0]
        n_y = source_dims[1] // downsampling_factor[1]
        target_dims = (n_x, n_y)
    elif target_dims is not None:
        pass
    else:
        raise Exception('please specify either downsampling_factor or target_dims')

    target_dims_pad = (target_dims[0] - 2 * padding, target_dims[1] - 2 * padding)

    # dvs_reshaped = reshape(dvs_mon, bot_left_corner=(-15, 10), top_right_corner=(200, 150))
    # dvs_mon = dvs_reshaped[0]
    # source_dims = dvs_reshaped[1]

    dvs_mon_ds = spatial_downsampling(dvs_mon, source_dims=source_dims, target_dims=target_dims_pad)
    dvs_mon_ds = add_padding(dvs_mon_ds, padding)

    dvs_plotter = Plotter2d(dvs_mon_ds, dims=target_dims)

    if plot:
        from brian2 import ms
        from pyqtgraph.Qt import QtGui  # , QtCore

        app = QtGui.QApplication.instance()
        if app is None:
            app = QtGui.QApplication(sys.argv)
        else:
            print('QApplication instance already exists: %s' % str(app))
        dt = 10 * ms
        imv = dvs_plotter.plot3d_on_off(plot_dt=dt, filtersize=100 * ms)
        imv.show()
        app.exec()

        # imv = dvs_plotter.plot3d(plot_dt=dt, filtersize=100 * ms)
        # imv.show()
        # app.exec()

    # dvs_plotter.xi
    # dvs_plotter.yi
    # spiketimes = np.asarray(dvs_plotter.t / dt, dtype=int)
    # spiketimes = spiketimes - np.min(spiketimes)

    return dvs_plotter
