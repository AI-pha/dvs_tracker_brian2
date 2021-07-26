import numpy as np
from brian2 import NeuronGroup, Synapses, SpikeMonitor, StateMonitor

from loihi_tools.distance_matrix import create_dist_mat_resize
from loihi_tools.prob_distributions import normal1d_density
from loihi_tools.loihi_equations import syn_eq, on_pre, add_clipping_to_NeuronGroup, loihi_eq_dict


def gaussian_weight_matrix(predims, postdims, sigma, weight, sigma_inh=None, weight_inh=0, threshold=0.1):
    dist_mat = create_dist_mat_resize(predims=predims, postdims=postdims)
    weights = weight * normal1d_density(dist_mat, sigma=sigma, normalized=False)
    if sigma_inh is not None:
        weights = weights + weight_inh * normal1d_density(dist_mat, sigma=sigma_inh, normalized=False)

    mask = np.abs(weights) > np.max(np.abs(weights)) * threshold
    weights = np.floor(weights)

    return weights, mask


def gaussian_weights_ij(predims, postdims, sigma, weight, sigma_inh=None, weight_inh=0, threshold=0.1):
    weights, mask = gaussian_weight_matrix(predims, postdims, sigma, weight,
                                           sigma_inh=sigma_inh, weight_inh=weight_inh, threshold=threshold)
    i, j = np.where(mask)
    weights = weights[mask]  # type: np.ndarray

    return i, j, weights


def activate_local_inh(params):
    if params['exc_inh_weight'] == 0 or params['inh_exc_weight'] == 0:
        return False
    else:
        return True


def setup_tracking_layers(params, sim_dt):
    numX_pad = params['numX'] + 2 * params['padding']
    numY_pad = params['numY'] + 2 * params['padding']

    # create exc and inh layers
    exc_group = NeuronGroup(numX_pad * numY_pad, **loihi_eq_dict, name='exc')
    inh_group = NeuronGroup(numX_pad * numY_pad, **loihi_eq_dict, name='inh')
    # create glInh population
    ginh_group = NeuronGroup(params['num_ginh'], **loihi_eq_dict, name='ginh')
    out_group = NeuronGroup(numX_pad * numY_pad, **loihi_eq_dict, name='out')
    oginh_group = NeuronGroup(params['num_ginh'], **loihi_eq_dict, name='oginh')

    add_clipping_to_NeuronGroup(exc_group)
    add_clipping_to_NeuronGroup(inh_group)
    add_clipping_to_NeuronGroup(ginh_group)
    add_clipping_to_NeuronGroup(out_group)
    add_clipping_to_NeuronGroup(oginh_group)

    exc_group.vThMant = params['exc_thresh'] // 64
    exc_group.tau_v = params['exc_tau']
    exc_group.tau_i = params['exc_i_tau']
    exc_group.i_const = params['exc_i_const']
    exc_group.ref_p = params['exc_refp']

    inh_group.vThMant = params['inh_thresh'] // 64
    inh_group.tau_v = params['inh_tau']
    inh_group.tau_i = params['inh_i_tau']
    inh_group.ref_p = params['inh_refp']

    ginh_thresh = params['ginh_thresh'] // 64
    ginh_group.vThMant = params['ginh_thresh'] // 64
    ginh_group.tau_v = params['ginh_tau']
    ginh_group.tau_i = params['ginh_i_tau']
    ginh_group.ref_p = params['ginh_refp']

    out_group.vThMant = params['exc_thresh'] // 64
    out_group.tau_v = params['exc_tau']
    out_group.tau_i = params['exc_i_tau']
    out_group.i_const = params['exc_i_const']
    out_group.ref_p = params['exc_refp']

    oginh_group.vThMant = params['ginh_thresh'] // 64
    oginh_group.tau_v = params['ginh_tau']
    oginh_group.tau_i = params['ginh_i_tau']
    oginh_group.ref_p = params['ginh_refp']

    # create wrapped gaussian weight matrices and masks
    if activate_local_inh(params):
        iExc2Exc, jExc2Exc, wExc2Exc = gaussian_weights_ij(predims=(numX_pad, numY_pad),
                                                           postdims=(numX_pad, numY_pad),
                                                           sigma=params['exc_exc_sigma'],
                                                           weight=params['exc_exc_weight'])

        iExc2Inh, jExc2Inh, wExc2Inh = gaussian_weights_ij(predims=(numX_pad, numY_pad),
                                                           postdims=(numX_pad, numY_pad),
                                                           sigma=params['exc_inh_sigma'],
                                                           weight=params['exc_inh_weight'])

        iInh2Exc, jInh2Exc, wInh2Exc = gaussian_weights_ij(predims=(numX_pad, numY_pad),
                                                           postdims=(numX_pad, numY_pad),
                                                           sigma=params['inh_exc_sigma'],
                                                           weight=params['inh_exc_weight'])
    else:
        iExc2Exc, jExc2Exc, wExc2Exc = gaussian_weights_ij(predims=(numX_pad, numY_pad),
                                                           postdims=(numX_pad, numY_pad),
                                                           sigma=params['exc_exc_sigma'],
                                                           weight=params['exc_exc_weight'],
                                                           sigma_inh=params['inh_exc_sigma'],
                                                           weight_inh=params['inh_exc_weight'])

    iOut2Out, jOut2Out, wOut2Out = gaussian_weights_ij(predims=(numX_pad, numY_pad),
                                                       postdims=(numX_pad, numY_pad),
                                                       sigma=params['out_out_sigma'],
                                                       weight=params['out_out_weight'],
                                                       sigma_inh=params['inh_out_sigma'],
                                                       weight_inh=params['inh_out_weight'])

    # create lateral connections for exc_group with a Gaussian kernel
    exc_exc = Synapses(exc_group, exc_group, model=syn_eq, on_pre=on_pre, name='exc_exc')
    exc_exc.connect(i=iExc2Exc, j=jExc2Exc)
    exc_exc.weight = wExc2Exc
    exc_exc.w_factor = 1

    out_out = Synapses(out_group, out_group, model=syn_eq, on_pre=on_pre, name='out_out')
    out_out.connect(i=iOut2Out, j=jOut2Out)
    out_out.weight = wOut2Out
    out_out.w_factor = 1

    exc_out = Synapses(exc_group, out_group, model=syn_eq, on_pre=on_pre, name='exc_out')
    exc_out.connect('i==j')
    exc_out.weight = params['exc_out_weight']
    exc_out.w_factor = 1

    if activate_local_inh(params):
        # create exc2inh connections
        exc_inh = Synapses(exc_group, inh_group, model=syn_eq, on_pre=on_pre, name='exc_inh')
        exc_inh.connect(i=iExc2Inh, j=jExc2Inh)
        exc_inh.weight = wExc2Inh
        exc_inh.w_factor = 1

        # create inh2exc connections
        inh_exc = Synapses(inh_group, exc_group, model=syn_eq, on_pre=on_pre, name='inh_exc')
        inh_exc.connect(i=iInh2Exc, j=jInh2Exc)
        inh_exc.weight = wInh2Exc
        inh_exc.w_factor = 1

    # create exc2glInh connections
    exc_glinh = Synapses(exc_group, ginh_group, model=syn_eq, on_pre=on_pre, name='exc_ginh')
    exc_glinh.connect(True, p=params['exc_glinh_connp'])
    exc_glinh.weight = params['exc_glinh_weight']
    exc_glinh.w_factor = 1

    # create glInh2exc connections
    glinh_exc = Synapses(ginh_group, exc_group, model=syn_eq, on_pre=on_pre, name='ginh_exc')
    glinh_exc.connect(True, p=params['exc_glinh_connp'])
    glinh_exc.weight = params['glinh_exc_weight']
    glinh_exc.w_factor = 1

    # create out2glInh connections
    out_oglinh = Synapses(out_group, oginh_group, model=syn_eq, on_pre=on_pre, name='out_oginh')
    out_oglinh.connect(True, p=params['exc_glinh_connp'])
    out_oglinh.weight = params['out_glinh_weight']
    out_oglinh.w_factor = 1

    # create glInh2out connections
    oglinh_out = Synapses(oginh_group, out_group, model=syn_eq, on_pre=on_pre, name='ginh_out')
    oglinh_out.connect(True, p=params['exc_glinh_connp'])
    oglinh_out.weight = params['glinh_out_weight']
    oglinh_out.w_factor = 1

    exc_spikemon = SpikeMonitor(exc_group, name='exc_spikes')
    out_spikemon = SpikeMonitor(out_group, name='out_spikes')
    ginh_spikemon = SpikeMonitor(ginh_group, name='ginh_spikes')

    exc_statemon = StateMonitor(exc_group, record=True, variables=['v', 'i_syn'],
                                name='exc_state', dt=5 * sim_dt)

    groups = [exc_group, ginh_group, exc_exc, exc_glinh, glinh_exc]
    groups.append([out_group, oginh_group, out_out, exc_out, out_oglinh, oglinh_out])
    groups.append([exc_spikemon, out_spikemon, ginh_spikemon, exc_statemon])

    return groups
