"""
Copyright © 2021 University of Zurich.
Created by : Alpha Renner (alpren@ini.uzh.ch)

This is a brian2 implementation of:
Renner, A., Evanusa, M., & Sandamirskaya, Y. (2019).
Event-based attention and tracking on neuromorphic hardware.
arXiv preprint arXiv:1907.04060.

The best parameters are a bit different from the ones in the paper, as there is a small difference between the chip
and the brian 2 equations. Also, the way, the global inhibition is connected was changed to make it more scalable on the
hardware.
There are many different ways to set the parameters, it is likely that there is a parameter setting that is more robust,
especially also for other datasets than the one I chose here.
"""

import datetime
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from brian2 import Synapses, SpikeGeneratorGroup, Network, defaultclock, ms, prefs, set_device, device, second, codegen

import dvs_tracker
from dvs_tools.load_data import load_data
from dvs_tools.load_data import load_frames
from dvs_tools.plot_wta import plot_wta
from dvs_tracker.tracking_layers import gaussian_weights_ij
from dvs_tracker.tracking_layers import setup_tracking_layers
from loihi_tools.loihi_equations import syn_eq, on_pre
from loihi_tools.loihi_monitor import LoihiMonitor
from loihi_tools.spikegenerators import delete_doublets
from loihi_tools.spikegenerators import gaussian2d_poisson_spikes

np.random.seed(42)

prefs.codegen.target = 'numpy'

run_as_standalone = False
if run_as_standalone:
    standaloneDir = os.path.expanduser('~/DVS_tracker_standalone')
    set_device('cpp_standalone', directory=standaloneDir, build_on_run=False)
    device.reinit()
    device.activate(directory=standaloneDir, build_on_run=False)

    prefs.devices.cpp_standalone.extra_make_args_unix = ["-j$(nproc)"]


def setupNetwork(params, backend):
    start_time = time.time()

    defaultclock.dt = SIM_DT
    print('create net ...')

    groups = setup_tracking_layers(params, SIM_DT)

    b2net = Network()
    b2net.add(groups)

    numX_pad = params['numX'] + 2 * params['padding']
    numY_pad = params['numY'] + 2 * params['padding']

    if backend == 'brian2':
        NUM_LOADED_ROWS = 3000000
        if params['inputfile'] == 'shapes':
            inputfile = os.path.join('./data', 'events_shapes.npz')

        dvs_plotter = load_data(inputfile=inputfile,
                                target_dims=(numX_pad, numY_pad), downsampling_factor=None,
                                num_rows=NUM_LOADED_ROWS, max_time=params['num_steps'] * SIM_DT / ms,
                                padding=params['padding'])

        spiketimes_on = params['att_timesteps'] + np.asarray(dvs_plotter.t[dvs_plotter.pol == 0] / SIM_DT, dtype=int)
        indices_on = dvs_plotter.i[dvs_plotter.pol == 0]

        spiketimes_off = params['att_timesteps'] + np.asarray(dvs_plotter.t[dvs_plotter.pol == 1] / SIM_DT, dtype=int)
        indices_off = dvs_plotter.i[dvs_plotter.pol == 1]

        spiketimes_on, indices_on = delete_doublets(spiketimes_on, indices_on)
        spiketimes_on = spiketimes_on * SIM_DT

        spiketimes_off, indices_off = delete_doublets(spiketimes_off, indices_off)
        spiketimes_off = spiketimes_off * SIM_DT

        att_indices, att_timesteps = gaussian2d_poisson_spikes(mean_index=(params['padding'] + params['att_x'],
                                                                           params['padding'] + params['att_y']),
                                                               std_index=0.4,
                                                               num_timesteps=params['att_timesteps'],
                                                               numX=numX_pad,
                                                               numY=numY_pad, frequency=params['att_frequency'])

        dvs_spikegen_att = SpikeGeneratorGroup(N=numX_pad * numY_pad, indices=att_indices,
                                               times=att_timesteps * SIM_DT, name='att')
        dvs_spikegen_on = SpikeGeneratorGroup(N=numX_pad * numY_pad, indices=indices_on, times=spiketimes_on,
                                              name='dvs_on')
        dvs_spikegen_off = SpikeGeneratorGroup(N=numX_pad * numY_pad, indices=indices_off, times=spiketimes_off,
                                               name='dvs_off')

        isge2Exc, jsge2Exc, wsge2Exc = gaussian_weights_ij(predims=(numX_pad, numY_pad),
                                                           postdims=(numX_pad, numY_pad),
                                                           sigma=params['input_sigma'], weight=params['sge_exc_weight'])

        isgi2Exc, jsgi2Exc, wsgi2Exc = gaussian_weights_ij(predims=(numX_pad, numY_pad),
                                                           postdims=(numX_pad, numY_pad),
                                                           sigma=params['input_sigma'], weight=params['sgi_exc_weight'])

        att_to_out = True
        if att_to_out:
            att_group = b2net["out"]
        else:
            att_group = b2net["exc"]

        sga_exc = Synapses(dvs_spikegen_att, att_group, model=syn_eq, on_pre=on_pre, name='sga_att')
        sga_exc.connect('i==j')
        sga_exc.weight = params['sge_exc_weight']
        sga_exc.w_factor = 1

        sge_exc = Synapses(dvs_spikegen_on, b2net["exc"], model=syn_eq, on_pre=on_pre, name='sge_exc')
        sge_exc.connect(i=isge2Exc, j=jsge2Exc)
        sge_exc.weight = wsge2Exc
        sge_exc.w_factor = 1

        sgi_exc = Synapses(dvs_spikegen_off, b2net["exc"], model=syn_eq, on_pre=on_pre, name='sgi_exc')
        sgi_exc.connect(i=isgi2Exc, j=jsgi2Exc)
        sgi_exc.weight = wsgi2Exc
        sgi_exc.w_factor = 1

        b2net.dvs_plotter = dvs_plotter
        b2net.add((sge_exc, sgi_exc, sga_exc, dvs_spikegen_on, dvs_spikegen_off, dvs_spikegen_att))

        return b2net


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Start a simulation with given parameters.',
    #                                  epilog="Simulation done!")
    # parser.add_argument('-t', type=str, help='num_steps', default=40000)
    # parsed_args = parser.parse_args()
    # num_steps = int(parsed_args.t)
    #
    # print('num_steps:', num_steps)

    num_steps = 15000

    backend = 'brian2'

    if backend == 'brian2':
        from dvs_tracker.dvs_tracking_parameters_brian import params

    do_plot = params['do_plot']
    SIM_DT = params['SIM_DT']
    numX_pad = params['numX'] + 2 * params['padding']
    numY_pad = params['numY'] + 2 * params['padding']

    params['num_steps'] = num_steps
    start_time = time.time()
    net = setupNetwork(params=params, backend=backend)

    print('setup net for', time.time() - start_time)

    if backend == 'brian2':
        if run_as_standalone:
            start_time = time.time()
            net.run(duration=num_steps * SIM_DT, report="stdout", report_period=10 * second,
                    namespace=None, profile=True, level=0)
            device.build(compile=False, run=False, directory=standaloneDir, clean=True, debug=False)
            compiler, args = codegen.cpp_prefs.get_compiler_and_args()
            device.compile_source(directory=standaloneDir, compiler=compiler, clean=True, debug=False)
            print('compiled net for', time.time() - start_time)

            print('running soon...')
            start_time = time.time()
            device.run(directory=standaloneDir, with_output=True, run_args=[])
        else:
            start_time = time.time()
            net.run(num_steps * SIM_DT)

        print('net run for', time.time() - start_time)

        print('number of synapses')
        for group in net:
            if 'Synapses' in str(group):
                print(group.name, len(group))

        results = {}

        exc_spike_i = net['exc_spikes'].i
        exc_spike_t = np.asarray(net['exc_spikes'].t / SIM_DT, dtype=int)
        exc_spikes = np.zeros((numX_pad * numY_pad, num_steps))
        exc_spikes[exc_spike_i, exc_spike_t] = 1
        results.update({'exc': LoihiMonitor(exc_spikes, 1)})

        dvs_plotter = net.dvs_plotter
        dvs_plotter.set_range((0 * second, (num_steps - 1) * SIM_DT))
        results.update({'inp': LoihiMonitor(None, 1, t=dvs_plotter.t / SIM_DT, i=dvs_plotter.i, pol=dvs_plotter.pol)})

        out_spike_i = net['out_spikes'].i
        out_spike_t = np.asarray(net['out_spikes'].t / SIM_DT, dtype=int)
        out_spikes = np.zeros((numX_pad * numY_pad, num_steps))
        out_spikes[out_spike_i, out_spike_t] = 1
        results.update({'out': LoihiMonitor(out_spikes, 1)})

        ginh_spikes = net['ginh_spikes']
        # plt.figure()
        # plt.plot(ginh_spikes.t, ginh_spikes.i, '.')

        resultfile = './results/spikes_brian2_' + params['experiment_name'] + params['inputfile'] + '_.npz'
        if not os.path.exists(os.path.dirname(resultfile)):
            os.makedirs(os.path.dirname(resultfile))
            print('created directory', os.path.dirname(resultfile), 'in',
                  os.path.abspath(os.path.dirname(resultfile)))

        np.savez(resultfile, **results)

        print(datetime.datetime.now())

        if do_plot:
            datapath = os.path.join(dvs_tracker.__path__[0], '..', 'data', 'shapes_translation')
            frames, frames_timestamps = load_frames(datapath=datapath)
            spikes_plotter = plot_wta(file=os.path.abspath(resultfile),
                                      dims_exc=(numX_pad, numY_pad),
                                      dt_ms=SIM_DT / ms, filtersize_ms=50,
                                      generate_gif=False,
                                      show_plot=True,
                                      frames=frames,
                                      frames_timestamps=frames_timestamps * second + params['att_timesteps'] * SIM_DT
                                      )

            import loihi_tools.plotter2d
            __file__ = loihi_tools.plotter2d.__file__  # This is a workaround to make Pool work from the pycharm console

            # plot population vector trajectory of output layer
            x_center_tracker, y_center_tracker = \
                spikes_plotter.calculate_pop_vector_trajectory(dt=50 * ms, plot=False,
                                                               frame_timestamps=frames_timestamps)

            # plot trajectory of peak in output layer
            denseifrs, denseisis, densetimes = spikes_plotter.get_dense_ifr(frame_timestamps=frames_timestamps)
            image3d = np.reshape(denseifrs, (denseifrs.shape[0], spikes_plotter.cols, spikes_plotter.rows))
            x_center_tracker_peak, y_center_tracker_peak = spikes_plotter.peaks_trajectories(image3d, threshold=0.95,
                                                                                   filter_sigma=(0, 3, 3),
                                                                                   plot=False)

            plt.figure()
            plt.plot(x_center_tracker,
                     y_center_tracker)
            plt.title('population vector trajectory')
            plt.show()

            plt.figure()
            plt.plot(x_center_tracker_peak[:, 0],
                     y_center_tracker_peak[:, 0])
            plt.title('peak trajectory')
            plt.show()
