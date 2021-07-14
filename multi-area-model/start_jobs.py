import json
import os
import shutil

from config import base_path
from multiarea_model.default_params import nested_update, sim_params
try:
    from multiarea_model.sumatra_helpers import register_record
    sumatra_found = True
except ImportError:
    sumatra_found = False


def start_job(label, data_path, data_folder_hash,
              sumatra=False, reason=None, tag=None):
    """
    Start job on a compute cluster.

    Parameters
    ----------

    label : str
        Simulation label identifying the simulation to be run.
        The function loads all necessary files from the subfolder
        identified by the label.
    """

    # Copy run_simulation script to simulation folder
    shutil.copy2(os.path.join(base_path, 'run_simulation.py'),
                 os.path.join(data_path, data_folder_hash, 'run_simulation.py'))

    # Load simulation parameters
    fn = os.path.join(data_path,
                      data_folder_hash,
                      'custom_params')
    with open(fn, 'r') as f:
        custom_params = json.load(f)
    nested_update(sim_params, custom_params['sim_params'])

    # Copy custom param file for each MPI process
    for i in range(sim_params['num_processes']):
        shutil.copy(fn, '_'.join((fn, str(i))))

    # Create folder for storing simulation output
    os.mkdir(os.path.join(data_path,
                          data_folder_hash,
                          'recordings'))

    # If chosen, register simulation to sumatra
    if sumatra:
        if sumatra_found:
            register_record(label, reason=reason, tag=tag)
        else:
            raise ImportWarning('Sumatra is not installed, so'
                                'cannot register simulation record.')