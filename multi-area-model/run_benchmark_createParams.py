import numpy as np
import os
import sys
import json
import nest

from multiarea_model import MultiAreaModel, MultiAreaModel_3
from config import base_path
from start_jobs import start_job
from figures.Schmidt2018_dyn.network_simulations import NEW_SIM_PARAMS

N_scaling = float(sys.argv[1])
num_processes = int(sys.argv[2])
t_sim = float(sys.argv[3])
K_scaling = float(sys.argv[4])
data_path = sys.argv[5]
data_folder_hash = sys.argv[6]
# Fig3: corresponds to figure 3 in schmidt et al. 2018: Groundstate
mam_state = sys.argv[7]
# Fig5: corresponds to figure 5 in schmidt et al. 2018: Metastable
rng_seed = int(sys.argv[8])
t_presim = float(sys.argv[9])
record_spikes = sys.argv[10] == 'True'

if mam_state == 'ground':
    figure = 'Fig3'
elif mam_state == 'metastable':
    figure = 'Fig5'
else:
    raise KeyError('No network state selected. Choose between "ground" and\
                   "metastable" state in the config file.')
network_params, _ = NEW_SIM_PARAMS[figure][0]

network_params['connection_params']['K_stable'] = os.path.join(
    base_path, 'K_stable.npy')
network_params['N_scaling'] = N_scaling
network_params['K_scaling'] = K_scaling
network_params['fullscale_rates'] = os.path.join(
    base_path, 'tests/fullscale_rates.json')

sim_params = {'t_sim': t_sim,
              't_presim': t_presim,
              'num_processes': num_processes,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False}}

if not record_spikes:
    sim_params['recording_dict']['areas_recorded'] = []

theory_params = {'dt': 0.1}

os.mkdir(os.path.join(data_path, data_folder_hash))

try:
    nest.version()
    NEST_version = '2'
except:
    nest.__version__
    NEST_version = '3'

if NEST_version == '2':
    print("NEST version 2.x\n")
    sim_params['master_seed'] = rng_seed
    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=True,
                       theory_spec=theory_params,
                       data_path=data_path,
                       data_folder_hash=data_folder_hash)
elif NEST_version == '3':
    print("NEST version 3.0\n")
    sim_params['rng_seed'] = rng_seed
    M = MultiAreaModel_3(network_params, simulation=True,
                         sim_spec=sim_params,
                         theory=True,
                         theory_spec=theory_params,
                         data_path=data_path,
                         data_folder_hash=data_folder_hash)

print(M.label)
print(M.simulation.label)

p, r = M.theory.integrate_siegert()
print("Mean-field theory predicts an average "
      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))

start_job(M.simulation.label, data_path, data_folder_hash)
