import json
import os

import nest


def logging(py_timers=None, memory_used=None):
    """
    Write runtime and memory for all MPI processes to file.
    """
    fn = os.path.join('data',
                      '_'.join(('logfile',
                                str(nest.Rank()))))
    with open(fn, 'w') as f:
        for key, val in nest.GetKernelStatus().items():
            f.write(key + ' ' + str(val) + '\n')
        if py_timers:
            for key, value in py_timers.items():
                f.write(key + ' ' + str(value) + '\n')
        if memory_used:
            for key, value in memory_used.items():
                f.write(key + ' ' + str(value) + '\n')


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
