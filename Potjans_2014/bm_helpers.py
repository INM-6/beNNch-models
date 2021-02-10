import numpy as np
import re
import json
import nest


def write_out_timer_data():
    """
    This function writes out measures taken with internal instrumentation of
    the code. If the instrumentation is activated, one usually measures delivery
    phase, update phase, communication phase, collocation phase and the total time.
    Probably only communication should be measured as this is the most
    interesting metric and because all other measurements might lead to performance
    degradation as they are implemented inside of omp barriers and make other
    threads wait. The extraction of the timer data assumes that there is a
    presimulation and simulation phase. The timer data is written to stdout,
    irregardless of which phase we are in. If there are two simulation calls of
    nest, i.e. a presim and sim phase, then one can savely assume that the second
    half of the stdout was written during the main simulation phase. Thus we only
    take the second half of matched patterns.
    """
    total_time_pattern = r'0] Total time'
    sim_time_pattern = r'0] Simulate time'
    update_time_pattern = r'0] Update time:'
    collocate_time_pattern = r'0] GatherSpikeData::collocate time:'
    communicate_time_pattern = r'GatherSpikeData::communicate time:'
    deliver_time_pattern = r'GatherSpikeData::deliver time:'

    stdout_file = 'stdout.app'

    total_time = grep_pattern_stdout(stdout_file, total_time_pattern)
    sim_time = grep_pattern_stdout(stdout_file, sim_time_pattern)
    update_timer = grep_pattern_stdout(stdout_file, update_time_pattern)
    collocate_timer = grep_pattern_stdout(stdout_file, collocate_time_pattern)
    communicate_timer = grep_pattern_stdout(stdout_file, communicate_time_pattern)
    deliver_timer = grep_pattern_stdout(stdout_file, deliver_time_pattern)

    _, _, total_phase_total_timer = split_presim_sim_mean(total_time)
    _, _, sim_phase_total_timer = split_presim_sim_mean(sim_time)
    _, _, update_phase_total_timer = split_presim_sim_mean(update_timer)
    _, _, collocate_phase_total_timer = split_presim_sim_mean(collocate_timer)
    _, _, communicate_phase_total_timer = split_presim_sim_mean(communicate_timer)
    _, _, deliver_phase_total_timer = split_presim_sim_mean(deliver_timer)

    outF = open("timer_data.txt", "w")
    outF.write('total_phase_total_timer: '+ str(total_phase_total_timer) + '\n')
    outF.write('sim_phase_total_timer: '+ str(sim_phase_total_timer) + '\n')
    outF.write('update_phase_total_timer: '+ str(update_phase_total_timer) + '\n')
    outF.write('collocate_phase_total_timer: '+ str(collocate_phase_total_timer) + '\n')
    outF.write('communicate_phase_total_timer: '+ str(communicate_phase_total_timer) + '\n')
    outF.write('deliver_phase_total_timer: '+ str(deliver_phase_total_timer) + '\n')
    outF.close()

def grep_pattern_stdout(stdout_file, pattern, typ='float'):
    sim_time = []
    with open(stdout_file) as f:
        for line in f:
            if re.findall(pattern, line):
                if typ == 'float':
                    # The number is always at location [3], we can just take
                    # this location and turn it into a float
                    time_tmp = float(line.split()[3])
                    sim_time.append(time_tmp)
                elif typ == 'int':
                    time_tmp = int(re.findall("\d+", line)[0])
                    sim_time.append(time_tmp)
    return np.array(sim_time)

def split_presim_sim_mean(time_tmp, split=True):
    # a is from the presim phase, b is presim and sim phase combined, c is only sim phase
    length = len(time_tmp)
    if split:
        a = time_tmp[:length//2]
        b = time_tmp[length//2:]
        a = a.mean()
        b = b.mean()
        c = b - a
        return a, b, c
    else:
        a = time_tmp.mean()
        return a,a,a


def write_out_KernelStatus():
    """ Write out the NEST Kernel Status """
    KernelStatus = nest.GetKernelStatus()
    with open('kernel_status.txt', 'w') as file:
        file.write(json.dumps(KernelStatus))

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

