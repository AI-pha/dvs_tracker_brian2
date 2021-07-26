#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright © 2021 University of Zurich.
Created by : Alpha Renner (alpren@ini.uzh.ch)
"""

from brian2 import ms

params = {}

params['experiment_name'] = '64x64_brian'
params['inputfile'] = 'shapes'
params['numX'] = 64
params['numY'] = 64
params['padding'] = 0

numX_padded = params['numX'] + 2 * params['padding']
numY_padded = params['numY'] + 2 * params['padding']

params['exc_exc_weight'] = 200
params['exc_inh_weight'] = 0  # There is no separate local inhibitory group
params['inh_exc_weight'] = -55

params['exc_glinh_weight'] = 5
params['glinh_exc_weight'] = -20

params['exc_out_weight'] = 850

params['out_out_weight'] = 235
params['inh_out_weight'] = -35

params['out_glinh_weight'] = 5
params['glinh_out_weight'] = -90

params['sge_exc_weight'] = 70
params['sgi_exc_weight'] = -50

params['exc_exc_sigma'] = 2.0
params['inh_exc_sigma'] = 4.0

params['out_out_sigma'] = 2.0
params['inh_out_sigma'] = 4.0

params['input_sigma'] = 1.5

params['exc_refp'] = 12
params['inh_refp'] = 12
params['ginh_refp'] = 7

params['exc_tau'] = 20
params['inh_tau'] = 20
params['ginh_tau'] = 20

params['exc_i_tau'] = 20
params['inh_i_tau'] = 20
params['ginh_i_tau'] = 20

params['exc_thresh'] = 64 * 10 * 64
params['inh_thresh'] = 64 * 10 * 64
params['ginh_thresh'] = 64 * 14 * 64

params['exc_glinh_connp'] = 0.6

params['exc_i_const'] = 0

params['num_ginh'] = 40

# Parameters of the initial "attention" input
params['att_timesteps'] = 300
params['att_frequency'] = 40
params['att_x'] = round(0.65 * params['numX'])
params['att_y'] = round(0.25 * params['numY'])

params['doProbes'] = False
params['do_plot'] = True
params['SIM_DT'] = 0.5 * ms
