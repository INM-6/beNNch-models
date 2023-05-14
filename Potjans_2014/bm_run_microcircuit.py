# -*- coding: utf-8 -*-
#
# bm_run_microcircuit.py

"""PyNEST Microcircuit: Run Benchmark Simulation
--------------------------------------------------

This is an example script for running the microcircuit model.
This version is adjusted for benchmark simulations. Since spikes are
usually not recorded in this scenario, the evaluation part with plotting of
'run_microcircuit.py' is not performed here.

"""

###############################################################################
# Import the necessary modules and start the time measurements.

from stimulus_params import stim_dict
from network_params import net_dict
from sim_params import sim_dict
from bm_helpers import logging, memory
import network
import nest
import time
time_start = time.time()

###############################################################################
# Initialize the network with simulation, network and stimulation parameters,
# then create and connect all nodes, and finally simulate.
# The times for a presimulation and the main simulation are taken
# independently. A presimulation is useful because the spike activity typically
# exhibits a startup transient. In benchmark simulations, this transient should
# be excluded from a time measurement of the state propagation phase. Besides,
# statistical measures of the spike activity should only be computed after the
# transient has passed.
#
# Benchmark: In contrast to run_microcircuit.py, some default simulation and
# network parameters are here overwritten.


sim_dict.update({
    't_presim': {model_time_presim},
    't_sim': {model_time_sim},
    'rec_dev': [{record_spikes}],
    'rng_seed': {rng_seed},
    'local_num_threads': {threads_per_task},
    'print_time': False,
    'kwds': {kwds}})

net_dict.update({
    'N_scaling': {scale_N},
    'K_scaling': {scale_K},
    'poisson_input': {poisson_input},
    'V0_type': {V0_type},
    'synapse_type': {synapse_type}})

py_timers = {}
memory_used = {}

memory_used['base_memory'] = memory()

t0 = time.time()
net = network.Network(sim_dict, net_dict, stim_dict)
t1 = time.time()
py_timers['py_time_network'] = t1 - t0

net.create()
t2 = time.time()
py_timers['py_time_create'] = t2 - t1
memory_used['node_memory'] = memory()

net.connect()
t3 = time.time()
py_timers['py_time_connect'] = t3 - t2
memory_used['network_memory'] = memory()

net.simulate(sim_dict['t_presim'])
t4 = time.time()
py_timers['py_time_presimulate'] = t4 - t3
memory_used['init_memory'] = memory()

net.simulate(sim_dict['t_sim'])
t5 = time.time()
py_timers['py_time_simulate'] = t5 - t4
memory_used['total_memory'] = memory()

###############################################################################
# Summarize time measurements. Rank 0 usually takes longest because of print
# calls.

print(
    '\nTimes of Rank {}:\n'.format(
        nest.Rank()) +
    '  Total time:          {:.3f} s\n'.format(
        py_timers['py_time_simulate']) +
    '  Time to initialize:  {:.3f} s\n'.format(
        py_timers['py_time_network']) +
    '  Time to create:      {:.3f} s\n'.format(
        py_timers['py_time_create']) +
    '  Time to connect:     {:.3f} s\n'.format(
        py_timers['py_time_connect']) +
    '  Time to presimulate: {:.3f} s\n'.format(
        py_timers['py_time_presimulate']) +
    '  Time to simulate:    {:.3f} s\n'.format(
        py_timers['py_time_simulate']))

###############################################################################
# Query the accumulated number of spikes on each rank.

local_spike_counter = net.get_local_spike_counter()
num_neurons = net.get_network_size()
rate = 1. * local_spike_counter / num_neurons / net.get_total_sim_time() * 1000
mem = memory()

print(
    'local_spike_counter: {}'.format(
        local_spike_counter))

print(
    'Number of neurons: {}'.format(
        num_neurons))

print(
    'Rate per rank: {}'.format(
        rate))

print(
    'memory: {}'.format(
        mem))

logging(py_timers=py_timers, memory_used=memory_used)
