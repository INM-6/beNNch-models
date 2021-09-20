import json
import os

import nest


def logging(py_timers=None, memory_used=None):
    """
    Write runtime and memory for all MPI processes to file.
    """
    metrics = [
        'num_connections',
        'local_spike_counter',
        'time_collocate_spike_data',
        'time_communicate_spike_data',
        'time_communicate_target_data',
        'time_deliver_spike_data',
        'time_gather_spike_data',
        'time_gather_target_data',
        'time_update',
        'time_communicate_prepare',
        'time_construction_connect',
        'time_construction_create',
        'time_simulate'
    ]

    fn = os.path.join('data',
                      '_'.join(('logfile',
                                str(nest.Rank()))))
    with open(fn, 'w') as f:
        for m in metrics:
            f.write(m + ' ' + str(nest.GetKernelStatus(m)) + '\n')
        if py_timers:
            for key, value in py_timers.items():
                f.write(key + ' ' + str(value) + '\n')
        if memory_used:
            for key, value in memory_used.items():
                f.write(key + ' ' + str(value) + '\n')


def write_out_KernelStatus(fname='kernel_status.txt'):
    """
    Writes out the NEST Kernel Status.

    Parameters
    ----------
    fname
        file name
    """
    KernelStatus = nest.GetKernelStatus()
    with open(fname, 'w') as f:
        f.write(json.dumps(KernelStatus))


def memory():
    """
    Use NEST's memory wrapper function to record used memory.
    """
    try:
        mem = nest.ll_api.sli_func('memory_thisjob')
    except AttributeError:
        mem = nest.sli_func('memory_thisjob')
    if isinstance(mem, dict):
        return mem['heap']
    else:
        return mem
