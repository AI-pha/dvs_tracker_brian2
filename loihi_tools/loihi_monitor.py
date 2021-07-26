import numpy as np


class LoihiMonitor:
    def __init__(self, spike_probe, dt, t=None, i=None, pol=None, brian2_monitor=None):

        if t is None:
            try:
                probedata = spike_probe.data
            except AttributeError:
                probedata = spike_probe

            indices_and_time = np.where(probedata)
            indices_and_time = np.asarray(indices_and_time)
            indices_and_time = indices_and_time[:, indices_and_time[1, :].argsort()]
            i, t = indices_and_time

        self.t = t * dt
        self.i = i
        self.pol = pol

        if brian2_monitor is not None:
            self.source = brian2_monitor.source
            self.record = brian2_monitor.record

    def savez(self, filename):
        """
        saves the monitor in a sparse way.
        only i,t are saved to an npz

        Args:
            filename (str): filename under which to save the data of the plotter object
        """
        np.savez_compressed(str(filename) + ".npz", i=self.i, t=self.t, pol=self.pol)
