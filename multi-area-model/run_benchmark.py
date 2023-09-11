import numpy as np
import os
import sys
import json
import nest
import glob

from multiarea_model import MultiAreaModel, MultiAreaModel_3

"""
Create parameters.
"""

data_path = sys.argv[1]
data_folder_hash = sys.argv[2]
mam_state = sys.argv[3]
# Fig3: corresponds to figure 3 in schmidt et al. 2018: Groundstate
# Fig5: corresponds to figure 5 in schmidt et al. 2018: Metastable

print("load simulation parameters\n")

# Load simulation parameters
fn = os.path.join(data_path,
                  data_folder_hash,
                  '_'.join(('custom_params', str(nest.Rank()))))
with open(fn, 'r') as f:
    custom_params = json.load(f)
    extra_params = {kwds}
    if extra_params:
        custom_params['sim_params'].update(extra_params)

print("Create network and simulate\n")

try:
    nest.version()
    NEST_version = '2'
except:
    nest.__version__
    NEST_version = '3'

if NEST_version == '2':
    M = MultiAreaModel('benchmark',
                       simulation=True,
                       sim_spec=custom_params['sim_params'],
                       data_path=data_path,
                       data_folder_hash=data_folder_hash)
elif NEST_version == '3':
    M = MultiAreaModel_3('benchmark',
                         simulation=True,
                         sim_spec=custom_params['sim_params'],
                         data_path=data_path,
                         data_folder_hash=data_folder_hash)
print("simulate\n")
M.simulation.simulate()

# delete copies of custom_params
if NEST_version == '2':
    custom_params_duplicates = glob.glob(os.path.join(
        data_path,
        data_folder_hash,
        'custom_params_*'))
    for duplicate in custom_params_duplicates:
        os.remove(duplicate)
