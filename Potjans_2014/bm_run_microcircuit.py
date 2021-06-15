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
from bm_helpers import write_out_timer_data, write_out_KernelStatus, memory
import network
import nest
import numpy as np
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
    'rng_seed': {seed},
    'local_num_threads': {threads_per_node},
    'print_time': False,
    'kwds': [{kwds}]})

net_dict.update({
    'N_scaling': {N_SCALING},
    'K_scaling': {K_SCALING},
    'poisson_input': {POISSON_INPUT},
    'V0_type': {V0_TYPE},
    'synapse_type': {SYNAPSE_TYPE}})

net = network.Network(sim_dict, net_dict, stim_dict)
time_network = time.time()

net.create()
time_create = time.time()

net.connect()
time_connect = time.time()

net.simulate(sim_dict['t_presim'])
time_presimulate = time.time()

net.simulate(sim_dict['t_sim'])
time_simulate = time.time()

###############################################################################
# Summarize time measurements. Rank 0 usually takes longest because of print
# calls.

print(
    '\nTimes of Rank {}:\n'.format(
        nest.Rank()) +
    '  Total time:          {:.3f} s\n'.format(
        time_simulate -
        time_start) +
    '  Time to initialize:  {:.3f} s\n'.format(
        time_network -
        time_start) +
    '  Time to create:      {:.3f} s\n'.format(
        time_create -
        time_network) +
    '  Time to connect:     {:.3f} s\n'.format(
        time_connect -
        time_create) +
    '  Time to presimulate: {:.3f} s\n'.format(
        time_presimulate -
        time_connect) +
    '  Time to simulate:    {:.3f} s\n'.format(
        time_simulate -
        time_presimulate))

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

write_out_KernelStatus()
write_out_timer_data()
