
import warnings

from brian2 import Equations

# neuron eq
loihi_eq = Equations('''
dv/dt = -v/(tau_v*dt)+i_syn/dt+i_const/dt : 1 (unless refractory)
di_syn/dt = -i_syn/(tau_i*dt) : 1
i_const : 1  (shared)
vThMant : 1 (shared)

tau_v  : 1 (shared)
tau_i  : 1 (shared)
threshold = vThMant * 64 : 1
ref_p : 1 (shared)
''')

loihi_eq_dict = {
    'model': loihi_eq,
    'threshold': 'v>threshold',
    'reset': 'v = 0',
    'refractory': 'ref_p*dt',
    'method': 'euler',
}

# synapse eq
syn_eq = '''weight :1
w_factor : 1 (shared)'''
on_pre = '''
i_syn_post += 64 * weight * w_factor
'''

loihi_syn_dict = {'model': syn_eq,
                  'on_pre': on_pre,
                  'method': 'euler'
                  }

def add_clipping_to_NeuronGroup(neurongroup):
    clip_v = 'v = clip(v,-2**23,2**23)'
    clip_i_syn = 'i_syn = clip(i_syn,-2**23,2**23)'
    neurongroup.run_regularly(clip_v)
    neurongroup.run_regularly(clip_i_syn)


def set_params(briangroup, params, ndargs=None):
    for par in params:
        if hasattr(briangroup, par):
            if ndargs is not None and par in ndargs:
                if ndargs[par] is None:
                    setattr(briangroup, par, params[par])
                else:
                    print(par, ndargs, ndargs[par])
                    setattr(briangroup, par, ndargs[par])
            else:
                setattr(briangroup, par, params[par])
        else:
            warnings.warn("Group " + str(briangroup.name) +
                          " has no state variable " + str(par) +
                          ", but you tried to set it with set_params")
