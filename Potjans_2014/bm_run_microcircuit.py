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
import network
import nest
import time
from pathlib import Path
from bennchutils.recorder import Recorder, yaml

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

def memory():
    """
    Use NEST's memory wrapper function to record used memory.
    """
    try:
        mem = nest.ll_api.sli_func("memory_thisjob")
    except AttributeError:
        mem = nest.sli_func("memory_thisjob")
    if isinstance(mem, dict):
        return mem["heap"]
    else:
        return mem


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

record = Recorder(
    fields_={
        "kernel_status": nest.GetKernelStatus,
        "memory": memory,
        "timestamp": time.time,
    }
)

def main():

    net = network.Network(sim_dict, net_dict, stim_dict)

    with record('create'):
        net.create()

    with record('connect'):
        net.connect()

    with record('warmup'):
        net.simulate(sim_dict['t_presim'])

    with record('simulate')
        net.simulate(sim_dict['t_sim'])


    ###############################################################################
    # Write out recorded data 
    data_path = Path('data') / f'data_{nest.Rank():03}.yaml'
    with data_path.open('w') as outfile:
        yaml.dump(record.model_dump(), outfile)

if __name__ == '__main__':
    main()
