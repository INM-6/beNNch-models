import numpy as np
import os
import sys
import json
import nest

from multiarea_model import MultiAreaModel, MultiAreaModel_3#, MultiAreaModel_rng
from config import base_path
from start_jobs import start_job
from figures.Schmidt2018_dyn.network_simulations import NEW_SIM_PARAMS

N_scaling = float(sys.argv[1])
num_processes = int(sys.argv[2])
t_sim = float(sys.argv[3])
K_scaling = float(sys.argv[4])
data_path = sys.argv[5]
data_folder_hash = sys.argv[6]
mam_state = sys.argv[7]  # Fig3: corresponds to figure 3 in schmidt et al. 2018: Groundstate
                         # Fig5: corresponds to figure 5 in schmidt et al. 2018: Metastable

network_params, _ = NEW_SIM_PARAMS[mam_state][0]

network_params['connection_params']['K_stable'] = os.path.join(base_path, 'K_stable.npy')
network_params['N_scaling'] = N_scaling
network_params['K_scaling'] = K_scaling
network_params['fullscale_rates'] = os.path.join(base_path, 'tests/fullscale_rates.json')

sim_params = {'t_sim': t_sim,
              'num_processes': num_processes,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False},
              'master_seed': 1}

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
    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=True,
                       theory_spec=theory_params,
                       data_path=data_path,
                       data_folder_hash=data_folder_hash)
elif NEST_version == '3':
    print("NEST version 3.0\n")
    M = MultiAreaModel_3(network_params, simulation=True,
                         sim_spec=sim_params,
                         theory=True,
                         theory_spec=theory_params,
                         data_path=data_path,
                         data_folder_hash=data_folder_hash)
# elif NEST_version == 'rng':
#     print("NEST version rng\n")
#     M = MultiAreaModel_rng(network_params, simulation=True,
#                            sim_spec=sim_params,
#                            theory=True,
#                            theory_spec=theory_params)

print(M.label)
print(M.simulation.label)

p, r = M.theory.integrate_siegert()
print("Mean-field theory predicts an average "
      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))

start_job(M.simulation.label, data_path, data_folder_hash)


