# -*- coding: utf-8 -*-
#
# hpc_benchmark.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.


"""
Random balanced network HPC benchmark
-------------------------------------

This script produces a balanced random network of `scale*11250` neurons in
which the excitatory-excitatory neurons exhibit STDP with
multiplicative depression and power-law potentiation. A mutual
equilibrium is obtained between the activity dynamics (low rate in
asynchronous irregular regime) and the synaptic weight distribution
(unimodal). The number of incoming connections per neuron is fixed
and independent of network size (indegree=11250).

This is the standard network investigated in [1]_, [2]_, [3]_.

A note on scaling
~~~~~~~~~~~~~~~~~

This benchmark was originally developed for very large-scale simulations on
supercomputers with more than 1 million neurons in the network and
11.250 incoming synapses per neuron. For such large networks, synaptic input
to a single neuron will be little correlated across inputs and network
activity will remain stable over long periods of time.

The original network size corresponds to a scale parameter of 100 or more.
In order to make it possible to test this benchmark script on desktop
computers, the scale parameter is set to 1 below, while the number of
11.250 incoming synapses per neuron is retained. In this limit, correlations
in input to neurons are large and will lead to increasing synaptic weights.
Over time, network dynamics will therefore become unstable and all neurons
in the network will fire in synchrony, leading to extremely slow simulation
speeds.

Therefore, the presimulation time is reduced to 50 ms below and the
simulation time to 250 ms, while we usually use 100 ms presimulation and
1000 ms simulation time.

For meaningful use of this benchmark, you should use a scale > 10 and check
that the firing rate reported at the end of the benchmark is below 10 spikes
per second.

References
~~~~~~~~~~

.. [1] Morrison A, Aertsen A, Diesmann M (2007). Spike-timing-dependent
       plasticity in balanced random networks. Neural Comput 19(6):1437-67
.. [2] Helias et al (2012). Supercomputers ready for use as discovery machines
       for neuroscience. Front. Neuroinform. 6:26
.. [3] Kunkel et al (2014). Spiking network simulation code for petascale
       computers. Front. Neuroinform. 8:78

"""

import numpy as np
import os
import time
import scipy.special as sp

import nest
# import nest.raster_plot

M_INFO = 10
M_ERROR = 30


###############################################################################
# Parameter section
# Define all relevant parameters: changes should be made here

params = {
    'nvp': {num_vps},                  # total number of virtual processes
    'num_threads': {threads_per_task}, # total number of threads per processes
    'scale': {scale},                  # scaling factor of the network size
                                       # total network size = scale*11250 neurons
    'simtime': {model_time_sim},       # total simulation time in ms
    'presimtime': {model_time_presim}, # simulation time until reaching equilibrium
    'dt': 0.1,                         # simulation step
    'record_spikes': {record_spikes},  # switch to record spikes of excitatory
                                       # neurons to file
    'rng_seed': {rng_seed},            # random number generator seed
    'path_name': '.',                  # path where all files will have to be written
    'log_file': 'logfile',             # naming scheme for the log files
}


def convert_synapse_weight(tau_m, tau_syn, C_m):
    """
    Computes conversion factor for synapse weight from mV to pA

    This function is specific to the leaky integrate-and-fire neuron
    model with alpha-shaped postsynaptic currents.

    """

    # compute time to maximum of V_m after spike input
    # to neuron at rest
    a = tau_m / tau_syn
    b = 1.0 / tau_syn - 1.0 / tau_m
    t_rise = 1.0 / b * (-lambertwm1(-np.exp(-1.0 / a) / a).real - 1.0 / a)

    v_max = np.exp(1.0) / (tau_syn * C_m * b) * (
        (np.exp(-t_rise / tau_m) - np.exp(-t_rise / tau_syn)) /
        b - t_rise * np.exp(-t_rise / tau_syn))
    return 1. / v_max

###############################################################################
# For compatibility with earlier benchmarks, we require a rise time of
# ``t_rise = 1.700759 ms`` and we choose ``tau_syn`` to achieve this for given
# ``tau_m``. This requires numerical inversion of the expression for ``t_rise``
# in ``convert_synapse_weight``. We computed this value once and hard-code
# it here.


tau_syn = 0.32582722403722841


brunel_params = {
    'NE': int(9000 * params['scale']),  # number of excitatory neurons
    'NI': int(2250 * params['scale']),  # number of inhibitory neurons

    'Nrec': 1000,  # number of neurons to record spikes from

    'model_params': {  # Set variables for iaf_psc_alpha
        'E_L': 0.0,  # Resting membrane potential(mV)
        'C_m': 250.0,  # Capacity of the membrane(pF)
        'tau_m': 10.0,  # Membrane time constant(ms)
        't_ref': 0.5,  # Duration of refractory period(ms)
        'V_th': 20.0,  # Threshold(mV)
        'V_reset': 0.0,  # Reset Potential(mV)
        # time const. postsynaptic excitatory currents(ms)
        'tau_syn_ex': tau_syn,
        # time const. postsynaptic inhibitory currents(ms)
        'tau_syn_in': tau_syn,
        'tau_minus': 30.0,  # time constant for STDP(depression)
        # V can be randomly initialized see below
        'V_m': 5.7  # mean value of membrane potential
    },

    ####################################################################
    # Note that Kunkel et al. (2014) report different values. The values
    # in the paper were used for the benchmarks on K, the values given
    # here were used for the benchmark on JUQUEEN.

    'randomize_Vm': True,
    'mean_potential': 5.7,
    'sigma_potential': 7.2,

    'delay': 1.5,  # synaptic delay, all connections(ms)

    # synaptic weight
    'JE': 0.14,  # peak of EPSP

    'sigma_w': 3.47,  # standard dev. of E->E synapses(pA)
    'g': -5.0,

    'stdp_params': {
        'delay': 1.5,
        'alpha': 0.0513,
        'lambda': 0.1,  # STDP step size
        'mu': 0.4,  # STDP weight dependence exponent(potentiation)
        'tau_plus': 15.0,  # time constant for potentiation
    },

    'eta': 1.685,  # scaling of external stimulus
    'filestem': params['path_name']
}

###############################################################################
# Function Section


def build_network():
    """Builds the network including setting of simulation and neuron
    parameters, creation of neurons and connections
    """

    tic = time.time()  # start timer on construction

    # unpack a few variables for convenience
    NE = brunel_params['NE']
    NI = brunel_params['NI']
    model_params = brunel_params['model_params']
    stdp_params = brunel_params['stdp_params']

    rng_seeds = list(
            range(
                params['rng_seed'] + 1 + params['nvp'],
                params['rng_seed'] + 1 + (2 * params['nvp'])
                )
            )
    grng_seed = params['rng_seed'] + params['nvp']
    # set global kernel parameters
    nest.SetKernelStatus({
        'local_num_threads': params['num_threads'],
        'resolution': params['dt'],
        'grng_seed': grng_seed,
        'rng_seeds': rng_seeds,
        'overwrite_files': True})
    nest.SetKernelStatus({kwds})

    nest.message(M_INFO, 'build_network', 'Creating excitatory population.')
    E_neurons = nest.Create('iaf_psc_alpha', NE, params=model_params)

    nest.message(M_INFO, 'build_network', 'Creating inhibitory population.')
    I_neurons = nest.Create('iaf_psc_alpha', NI, params=model_params)

    if brunel_params['randomize_Vm']:
        nest.message(M_INFO, 'build_network',
                     'Randomzing membrane potentials.')

        seed = nest.GetKernelStatus(
            'rng_seeds')[-1] + 1 + nest.GetStatus([0], 'vp')[0]
        rng = np.random.RandomState(seed=seed)

        for node in get_local_nodes(E_neurons):
            nest.SetStatus([node],
                           {'V_m': rng.normal(
                               brunel_params['mean_potential'],
                               brunel_params['sigma_potential'])})

        for node in get_local_nodes(I_neurons):
            nest.SetStatus([node],
                           {'V_m': rng.normal(
                               brunel_params['mean_potential'],
                               brunel_params['sigma_potential'])})

    # number of incoming excitatory connections
    CE = int(1. * NE / params['scale'])
    # number of incomining inhibitory connections
    CI = int(1. * NI / params['scale'])

    nest.message(M_INFO, 'build_network',
                 'Creating excitatory stimulus generator.')

    # Convert synapse weight from mV to pA
    conversion_factor = convert_synapse_weight(
        model_params['tau_m'], model_params['tau_syn_ex'], model_params['C_m'])
    JE_pA = conversion_factor * brunel_params['JE']

    nu_thresh = model_params['V_th'] / (
        CE * model_params['tau_m'] / model_params['C_m'] *
        JE_pA * np.exp(1.) * tau_syn)
    nu_ext = nu_thresh * brunel_params['eta']

    E_stimulus = nest.Create('poisson_generator', 1, {
                             'rate': nu_ext * CE * 1000.})

    nest.message(M_INFO, 'build_network',
                 'Creating excitatory spike recorder.')

    if params['record_spikes']:
        detector_label = os.path.join(
            brunel_params['filestem'],
            'alpha_' + str(stdp_params['alpha']) + '_spikes')
        E_detector = nest.Create('spike_detector', 1, {
            'withtime': True, 'to_file': True, 'label': detector_label})

    BuildNodeTime = time.time() - tic
    node_memory = str(memory_thisjob())

    tic = time.time()

    nest.SetDefaults('static_synapse_hpc', {'delay': brunel_params['delay']})
    nest.CopyModel('static_synapse_hpc', 'syn_ex',
                   {'weight': JE_pA})
    nest.CopyModel('static_synapse_hpc', 'syn_in',
                   {'weight': brunel_params['g'] * JE_pA})

    stdp_params['weight'] = JE_pA
    nest.SetDefaults('stdp_pl_synapse_hom_hpc', stdp_params)

    nest.message(M_INFO, 'build_network', 'Connecting stimulus generators.')

    # Connect Poisson generator to neuron

    nest.Connect(E_stimulus, E_neurons, {'rule': 'all_to_all'},
                 {'model': 'syn_ex'})
    nest.Connect(E_stimulus, I_neurons, {'rule': 'all_to_all'},
                 {'model': 'syn_ex'})

    nest.message(M_INFO, 'build_network',
                 'Connecting excitatory -> excitatory population.')

    nest.Connect(E_neurons, E_neurons,
                 {'rule': 'fixed_indegree', 'indegree': CE,
                  'autapses': False, 'multapses': True},
                 {'model': 'stdp_pl_synapse_hom_hpc'})

    nest.message(M_INFO, 'build_network',
                 'Connecting inhibitory -> excitatory population.')

    nest.Connect(I_neurons, E_neurons,
                 {'rule': 'fixed_indegree', 'indegree': CI,
                  'autapses': False, 'multapses': True},
                 {'model': 'syn_in'})

    nest.message(M_INFO, 'build_network',
                 'Connecting excitatory -> inhibitory population.')

    nest.Connect(E_neurons, I_neurons,
                 {'rule': 'fixed_indegree', 'indegree': CE,
                  'autapses': False, 'multapses': True},
                 {'model': 'syn_ex'})

    nest.message(M_INFO, 'build_network',
                 'Connecting inhibitory -> inhibitory population.')

    nest.Connect(I_neurons, I_neurons,
                 {'rule': 'fixed_indegree', 'indegree': CI,
                  'autapses': False, 'multapses': True},
                 {'model': 'syn_in'})

    if params['record_spikes']:
        local_neurons = list(get_local_nodes(E_neurons))

        if len(local_neurons) < brunel_params['Nrec']:
            nest.message(
                M_ERROR, 'build_network',
                """Spikes can only be recorded from local neurons, but the
                number of local neurons is smaller than the number of neurons
                spikes should be recorded from. Aborting the simulation!""")
            exit(1)

        nest.message(M_INFO, 'build_network', 'Connecting spike detectors.')
        nest.Connect(local_neurons[:brunel_params['Nrec']], E_detector,
                     'all_to_all', 'static_synapse_hpc')

    # read out time used for building
    BuildEdgeTime = time.time() - tic
    network_memory = str(memory_thisjob())

    d = {'py_time_create': BuildNodeTime,
         'py_time_connect': BuildEdgeTime,
         'node_memory': node_memory,
         'network_memory': network_memory}
    recorders = E_detector if params['record_spikes'] else None

    return d, recorders


def run_simulation():
    """Performs a simulation, including network construction"""

    nest.ResetKernel()
    nest.set_verbosity(M_INFO)

    base_memory = str(memory_thisjob())

    build_dict, sr = build_network()

    tic = time.time()

    nest.Prepare()

    InitTime = time.time() - tic
    init_memory = str(memory_thisjob())

    tic = time.time()

    nest.Run(params['presimtime'])

    PresimTime = time.time() - tic
    presim_memory = str(memory_thisjob())

    tic = time.time()

    nest.Run(params['simtime'])

    SimCPUTime = time.time() - tic
    total_memory = str(memory_thisjob())

    nest.Cleanup()

    average_rate = 0.0
    if params['record_spikes']:
        average_rate = compute_rate(sr)

    d = {'py_time_network_prepare': InitTime,
         'py_time_presimulate': PresimTime,
         'py_time_simulate': SimCPUTime,
         'base_memory': base_memory,
         'init_memory': init_memory,
         'presim_memory': presim_memory,
         'total_memory': total_memory,
         'average_rate': average_rate}
    d.update(build_dict)
    d.update(nest.GetKernelStatus())
    print(d)

    fn = '{fn}_{rank}.dat'.format(fn=params['log_file'], rank=nest.Rank())
    with open(fn, 'w') as f:
        for key, value in d.items():
            f.write(key + ' ' + str(value) + '\n')


def compute_rate(sr):
    """Compute local approximation of average firing rate

    This approximation is based on the number of local nodes, number
    of local spikes and total time. Since this also considers devices,
    the actual firing rate is usually underestimated.

    """

    n_local_spikes = nest.GetStatus(sr, 'n_events')[0]
    n_local_neurons = brunel_params['Nrec']
    simtime = params['simtime']
    return 1. * n_local_spikes / (n_local_neurons * simtime) * 1e3


def memory_thisjob():
    """Wrapper to obtain current memory usage"""
    nest.sr('memory_thisjob')
    return nest.spp()


def lambertwm1(x):
    """Wrapper for LambertWm1 function"""
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


def get_local_nodes(nodes):
    """Generator for efficient looping over local nodes
    Assumes nodes is a continous list of gids [1, 2, 3, ...], e.g., as
    returned by Create. Only works for nodes with proxies, i.e.,
    regular neurons.
    """

    i = 0
    while i < len(nodes):
        if nest.GetStatus([nodes[i]], 'local')[0]:
            yield nodes[i]
        i += 1


if __name__ == '__main__':
    run_simulation()
