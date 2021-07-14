import numpy as np
import re
import json
import nest


def write_out_timer_data(fname='timer_data.txt'):
    """
    Extracts timer data from the NEST KernelStatus and writes
    it into a single file to be later read in by JUBE.

    Parameters
    ----------
    fname
        file name
    """
    metrics = ['time_collocate_spike_data',
               'time_communicate_spike_data',
               'time_communicate_target_data',
               'time_deliver_spike_data',
               'time_gather_spike_data',
               'time_gather_target_data',
               'time_update',
               'time_communicate_prepare',
               'time_construction_connect',
               'time_construction_create',
               'time_simulate']

    d = nest.GetKernelStatus()
    with open(fname, 'w') as f:
        for m in metrics:
            if m in d:
                f.write(m + ' ' + str(d[m]) + '\n')


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
