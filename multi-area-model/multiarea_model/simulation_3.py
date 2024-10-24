
"""
multiarea_model
==============

Simulation class of the multi-area model of macaque visual vortex by
Schmidt et al. (2018).


Classes
-------
Simulation : Loads a parameter file that specifies simulation
parameters for a simulation of the instance of the model. A simulation
is identified by a unique hash label.

"""

import json
import nest
import numpy as np
import os
import pprint
import shutil
import time

from .analysis_helpers import _load_npy_to_dict, model_iter
from config import base_path
from copy import deepcopy
from .default_params_3 import nested_update, sim_params
from .default_params_3 import check_custom_params
from dicthash import dicthash
from .multiarea_helpers import extract_area_dict, create_vector_mask
try:
    from .sumatra_helpers import register_runtime
    sumatra_found = True
except ImportError:
    sumatra_found = False


class Simulation:
    def __init__(self, network, sim_spec, data_path=None, data_folder_hash=None):
        """
        Simulation class.
        An instance of the simulation class with the given parameters.
        Can be created as a member class of a multiarea_model instance
        or standalone.

        Parameters
        ----------
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network to be simulated.
        params : dict
            custom simulation parameters that overwrite the
            default parameters defined in default_params.py
        """
        print('GIT: ({})'.format(nest.__version__))

        self.data_folder_hash = data_folder_hash
        self.params = deepcopy(sim_params)
        if isinstance(sim_spec, dict):
            check_custom_params(sim_spec, self.params)
            self.custom_params = sim_spec
        else:
            fn = os.path.join(data_path,
                              sim_spec,
                              '_'.join(('custom_params',
                                        sim_spec)))
            with open(fn, 'r') as f:
                self.custom_params = json.load(f)['sim_params']

        nested_update(self.params, self.custom_params)

        self.network = network
        self.label = dicthash.generate_hash_from_dict({'params': self.params,
                                                       'network_label': self.network.label})

        print("Simulation label: {}".format(self.label))
        self.data_dir = os.path.join(data_path, self.data_folder_hash)
        try:
            os.mkdir(self.data_dir)
            os.mkdir(os.path.join(self.data_dir, 'recordings'))
        except OSError:
            pass
        self.copy_files()
        print("Copied files.")
        d = {'sim_params': self.custom_params,
             'network_params': self.network.custom_params,
             'network_label': self.network.label}
        with open(os.path.join(self.data_dir,
                               'custom_params'), 'w') as f:
            json.dump(d, f)
        print("Initialized simulation class.")

        self.areas_simulated = self.params['areas_simulated']
        self.areas_recorded = self.params['recording_dict']['areas_recorded']
        self.pre_T = self.params['t_presim']
        self.T = self.params['t_sim']

        self.time_create = 0
        self.time_connect_area = 0
        self.time_connect_cc = 0

        self.detailed_timers = 'time_communicate_spike_data' in nest.GetKernelStatus().keys()

    def __eq__(self, other):
        # Two simulations are equal if the simulation parameters and
        # the simulated networks are equal.
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        s = "Simulation {} of network {} with parameters:".format(self.label, self.network.label)
        s += pprint.pformat(self.params, width=1)
        return s

    def copy_files(self):
        """
        Copy all relevant files for the simulation to its data directory.
        """
        files = [os.path.join('multiarea_model',
                              'data_multiarea',
                              'Model.py'),
                 os.path.join('multiarea_model',
                              'data_multiarea',
                              'VisualCortex_Data.py'),
                 os.path.join('multiarea_model',
                              'multiarea_model.py'),
                 os.path.join('multiarea_model',
                              'simulation.py'),
                 os.path.join('multiarea_model',
                              'default_params.py')]
        if self.network.params['connection_params']['replace_cc_input_source'] is not None:
            fs = self.network.params['connection_params']['replace_cc_input_source']
            if '.json' in fs:
                files.append(fs)
            else:  # Assume that the cc input is stored in one npy file per population
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                for it in fn_iter:
                    fp_it = (fs,) + it
                    fp_ = '{}.npy'.format('-'.join(fp_it))
                    files.append(fp_)
        for f in files:
            shutil.copy2(os.path.join(base_path, f),
                         self.data_dir)

    def prepare(self):
        """
        Prepare NEST Kernel.
        """
        nest.ResetKernel()
        rng_seed = self.params['rng_seed']
        num_processes = self.params['num_processes']
        local_num_threads = self.params['local_num_threads']
        vp = num_processes * local_num_threads
        nest.SetKernelStatus({'resolution': self.params['dt'],
                              'total_num_virtual_procs': vp,
                              'overwrite_files': True,
                              'data_path': os.path.join(self.data_dir, 'recordings'),
                              'print_time': False,
                              'rng_seed': rng_seed,
                              'spike_buffer_grow_extra': self.params['spike_buffer_grow_extra'],
                              'spike_buffer_shrink_limit': self.params['spike_buffer_shrink_limit'],
                              'spike_buffer_shrink_spare': self.params['spike_buffer_shrink_spare']})

        # nest.set_verbosity('M_INFO')

        nest.SetDefaults(self.network.params['neuron_params']['neuron_model'],
                         self.network.params['neuron_params']['single_neuron_dict'])

    def create_recording_devices(self):
        """
        Create devices for all populations. Depending on the
        configuration, this will create:
        - spike recorder
        - voltmeter
        """
        if len(self.areas_recorded) != 0:
            self.spike_recorder = nest.Create('spike_recorder')
            status_dict = deepcopy(self.params['recording_dict']['spike_dict'])
            label = '-'.join((self.label,
                              status_dict['label']))
            status_dict.update({'label': label})
            self.spike_recorder.set(status_dict)

        if self.params['recording_dict']['record_vm']:
            self.voltmeter = nest.Create('voltmeter')
            status_dict = self.params['recording_dict']['vm_dict']
            label = '-'.join((self.label,
                              status_dict['label']))
            status_dict.update({'label': label})
            self.voltmeter.set(status_dict)

    def create_areas(self):
        """
        Create all areas with their populations and internal connections.
        """
        self.areas = []
        for area_name in self.areas_simulated:
            a = Area(self, self.network, area_name)
            self.areas.append(a)
            self.time_create += a.time_create
            self.time_connect_area += a.time_connect_area
            print("Memory after {0} : {1:.2f} MB".format(area_name, self.memory() / 1024.))

    def cortico_cortical_input(self):
        """
        Create connections between areas.
        """
        replace_cc = self.network.params['connection_params']['replace_cc']
        replace_non_simulated_areas = self.network.params['connection_params'][
            'replace_non_simulated_areas']
        if self.network.params['connection_params']['replace_cc_input_source'] is None:
            replace_cc_input_source = None
        else:
            replace_cc_input_source = os.path.join(self.data_dir,
                                                   self.network.params['connection_params'][
                                                       'replace_cc_input_source'])

        if not replace_cc and set(self.areas_simulated) != set(self.network.area_list):
            if replace_non_simulated_areas == 'het_current_nonstat':
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                non_simulated_cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
            elif replace_non_simulated_areas == 'het_poisson_stat':
                fn = self.network.params['connection_params']['replace_cc_input_source']
                with open(fn, 'r') as f:
                    non_simulated_cc_input = json.load(f)
            elif replace_non_simulated_areas == 'hom_poisson_stat':
                non_simulated_cc_input = {source_area_name:
                                          {source_pop:
                                           self.network.params['input_params']['rate_ext']
                                           for source_pop in
                                           self.network.structure[source_area_name]}
                                          for source_area_name in self.network.area_list}
            else:
                raise KeyError("Please define a valid method to"
                               " replace non-simulated areas.")

        if replace_cc == 'het_current_nonstat':
            fn_iter = model_iter(mode='single', areas=self.network.area_list)
            cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
        elif replace_cc == 'het_poisson_stat':
            with open(self.network.params['connection_params'][
                    'replace_cc_input_source'], 'r') as f:
                cc_input = json.load(f)
        elif replace_cc == 'hom_poisson_stat':
            cc_input = {source_area_name:
                        {source_pop:
                         self.network.params['input_params']['rate_ext']
                         for source_pop in
                         self.network.structure[source_area_name]}
                        for source_area_name in self.network.area_list}

        t0 = time.time()
        # Connections between simulated areas are not replaced
        if not replace_cc:
            for target_area in self.areas:
                # Loop source area though complete list of areas
                for source_area_name in self.network.area_list:
                    if target_area.name != source_area_name:
                        # If source_area is part of the simulated network,
                        # connect it to target_area
                        if source_area_name in self.areas:
                            source_area = self.areas[self.areas.index(source_area_name)]
                            connect(self,
                                    target_area,
                                    source_area)
                        # Else, replace the input from source_area with the
                        # chosen method
                        else:
                            target_area.create_additional_input(replace_non_simulated_areas,
                                                                source_area_name,
                                                                non_simulated_cc_input[
                                                                    source_area_name])
        # Connections between all simulated areas are replaced
        else:
            for target_area in self.areas:
                for source_area in self.areas:
                    if source_area != target_area:
                        target_area.create_additional_input(replace_cc,
                                                            source_area.name,
                                                            cc_input[source_area.name])
        t1 = time.time()
        self.time_connect_cc = t1 - t0

    def simulate(self):
        """
        Create the network and execute simulation.
        Record used memory and wallclock time.
        """
        t0 = time.time()
        self.base_memory = self.memory()
        self.prepare()
        t1 = time.time()
        self.time_kernel_prepare = t1 - t0
        print("Prepared simulation in {0:.2f} seconds.".format(self.time_kernel_prepare))

        self.create_recording_devices()
        self.create_areas()
        t2 = time.time()
        self.time_network_local = t2 - t1
        print("Created areas and internal connections in {0:.2f} seconds.".format(
            self.time_network_local))

        self.cortico_cortical_input()
        t3 = time.time()
        self.network_memory = self.memory()
        self.time_network_global = t3 - t2
        print("Created cortico-cortical connections in {0:.2f} seconds.".format(
            self.time_network_global))

        self.save_network_gids()

        print("Network size:", nest.GetKernelStatus('network_size'))
        print("Saved network in {0:2f} seconds.".format(time.time() - t3))

        t4 = time.time()
        nest.Prepare()
        self.time_network_prepare = time.time() - t4
        print("Network preparation time in {0:.2f} seconds.".format(self.time_network_prepare))

        t5 = time.time()
        nest.Run(self.pre_T)
        self.time_presimulate = time.time() - t5
        self.init_memory = self.memory()
        if self.detailed_timers:
            self.logging_presim()
        print("Presimulation time in {0:.2f} seconds.".format(self.time_presimulate))

        t6 = time.time()
        nest.Run(self.T)
        self.time_simulate = time.time() - t6

        self.total_memory = self.memory()
        print("Simulated network in {0:.2f} seconds.".format(self.time_simulate))
        self.logging()

    def memory(self):
        """
        Use NEST's memory wrapper function to record used memory.
        """
        mem = nest.ll_api.sli_func('memory_thisjob')
        if isinstance(mem, dict):
            return mem['heap']
        else:
            return mem

    def logging_presim(self):
        timer_keys = ['time_collocate_spike_data',
                      'time_communicate_spike_data',
                      'time_deliver_spike_data',
                      'time_gather_spike_data',
                      'time_update',
                      'time_simulate'
                      ]
        values = nest.GetKernelStatus(timer_keys)

        self.presim_timers = dict(zip(timer_keys, values))

        fn = os.path.join(self.data_dir,
                          'recordings',
                          '_'.join((self.label,
                                    'logfile',
                                    str(nest.Rank()))))

        with open(fn, 'w') as f:
            for idx, value in enumerate(values):
                f.write('presim_' + timer_keys[idx] + ' ' + str(value) + '\n')
            f.write('presim_local_spike_counter' + ' ' + str(nest.GetKernelStatus('local_spike_counter')) + '\n')

    def logging(self):
        """
        Write runtime and memory for all MPI processes to file.
        """
        d = {'py_time_kernel_prepare': self.time_kernel_prepare,
             'py_time_network_local': self.time_network_local,
             'py_time_network_global': self.time_network_global,
             'py_time_presimulate': self.time_presimulate,
             'py_time_network_prepare': self.time_network_prepare,
             'py_time_simulate': self.time_simulate,
             'py_time_create': self.time_create,
             'py_time_connect': self.time_connect_area + self.time_connect_cc + self.time_network_prepare,
             'py_time_connect_area': self.time_connect_area,
             'py_time_connect_cc': self.time_connect_cc,
             'base_memory': self.base_memory,
             'network_memory': self.network_memory,
             'init_memory': self.init_memory,
             'total_memory': self.total_memory}
        d.update(nest.GetKernelStatus())

        if self.detailed_timers:
            # subtract presim timers from simtime timers
            for key in self.presim_timers.keys():
                d[key] -= self.presim_timers[key]
            
        print(d)

        fn = os.path.join(self.data_dir,
                          'recordings',
                          '_'.join((self.label,
                                    'logfile',
                                    str(nest.Rank()))))
        with open(fn, 'a') as f:
            for key, value in d.items():
                f.write(key + ' ' + str(value) + '\n')

    def save_network_gids(self):
        with open(os.path.join(self.data_dir,
                               'recordings',
                               'network_gids.txt'), 'w') as f:
            for area in self.areas:
                for pop in self.network.structure[area.name]:
                    first_id = area.gids[pop][0].get()['global_id']
                    last_id = area.gids[pop][-1].get()['global_id']
                    f.write("{area},{pop},{g0},{g1}\n".format(area=area.name,
                                                              pop=pop,
                                                              g0=first_id,
                                                              g1=last_id))

    def register_runtime(self):
        if sumatra_found:
            register_runtime(self.label)
        else:
            raise ImportWarning('Sumatra is not installed, the '
                                'runtime cannot be registered.')


class Area:
    def __init__(self, simulation, network, name):
        """
        Area class.
        This class encapsulates a single area of the model.
        It creates all populations and the intrinsic connections between them.
        It provides an interface to allow connecting the area to other areas.

        Parameters
        ----------
        simulation : simulation
           An instance of the simulation class that specifies the
           simulation that the area is part of.
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network the area is part of.
        name : str
            Name of the area.
        """

        self.name = name
        self.simulation = simulation
        self.network = network
        self.neuron_numbers = network.N[name]
        self.synapses = extract_area_dict(network.synapses,
                                          network.structure,
                                          self.name,
                                          self.name)
        self.W = extract_area_dict(network.W,
                                   network.structure,
                                   self.name,
                                   self.name)
        self.W_sd = extract_area_dict(network.W_sd,
                                      network.structure,
                                      self.name,
                                      self.name)
        self.populations = network.structure[name]

        self.external_synapses = {}
        for pop in self.populations:
            self.external_synapses[pop] = self.network.K[self.name][pop]['external']['external']

        t0 = time.time()
        self.create_populations()
        t1 = time.time()
        self.time_create = t1 - t0
        self.connect_devices()
        self.connect_populations()
        self.time_connect_area = time.time() - t1
        print("Rank {}: created area {} with {} local nodes".format(nest.Rank(),
                                                                    self.name,
                                                                    self.num_local_nodes))

    def __str__(self):
        s = "Area {} with {} neurons.".format(
            self.name, int(self.neuron_numbers['total']))
        return s

    def __eq__(self, other):
        # If other is an instance of area, it should be the exact same
        # area This opens the possibility to have multiple instance of
        # one cortical areas
        if isinstance(other, Area):
            return self.name == other.name and self.gids == other.gids
        elif isinstance(other, str):
            return self.name == other

    def create_populations(self):
        """
        Create all populations of the area.
        """
        self.gids = {}
        self.num_local_nodes = 0
        for pop in self.populations:
            gid = nest.Create(self.network.params['neuron_params']['neuron_model'],
                              int(self.neuron_numbers[pop]))
            mask = create_vector_mask(self.network.structure, areas=[self.name], pops=[pop])
            I_e = self.network.add_DC_drive[mask][0]
            if not self.network.params['input_params']['poisson_input']:
                K_ext = self.external_synapses[pop]
                W_ext = self.network.W[self.name][pop]['external']['external']
                tau_syn = self.network.params['neuron_params']['single_neuron_dict']['tau_syn_ex']
                DC = K_ext * W_ext * tau_syn * 1.e-3 * \
                    self.network.params['rate_ext']
                I_e += DC
            gid.set({'I_e': I_e})

            # Store GIDCollection of each population
            self.gids[pop] = gid

            # Initialize membrane potentials
            # This could also be done after creating all areas, which
            # might yield better performance. Has to be tested.
            gid.set({'V_m':
                     nest.random.normal(self.network.params['neuron_params']['V0_mean'],
                                       self.network.params['neuron_params']['V0_sd'])})

    def connect_populations(self):
        """
        Create connections between populations.
        """
        connect(self.simulation,
                self,
                self)

    def connect_devices(self):
        if self.name in self.simulation.params['recording_dict']['areas_recorded']:
            for pop in self.populations:
                # Always record spikes from all neurons to get correct
                # statistics
                nest.Connect(self.gids[pop],
                             self.simulation.spike_recorder)

        if self.simulation.params['recording_dict']['record_vm']:
            for pop in self.populations:
                nrec = int(self.simulation.params['recording_dict']['Nrec_vm_fraction'] *
                           self.neuron_numbers[pop])
                nest.Connect(self.simulation.voltmeter,
                             self.gids[pop][:nrec])
        if self.network.params['input_params']['poisson_input']:
            self.poisson_generators = []
            for pop in self.populations:
                K_ext = self.external_synapses[pop]
                W_ext = self.network.W[self.name][pop]['external']['external']
                pg = nest.Create('poisson_generator')
                pg.set({'rate': self.network.params['input_params']['rate_ext'] * K_ext})
                syn_spec = {'weight': W_ext}
                nest.Connect(pg,
                             self.gids[pop],
                             syn_spec=syn_spec)
                self.poisson_generators.append(pg[0])

    def create_additional_input(self, input_type, source_area_name, cc_input):
        """
        Replace the input from a source area by the chosen type of input.

        Parameters
        ----------
        input_type : str, {'het_current_nonstat', 'hom_poisson_stat',
                           'het_poisson_stat'}
            Type of input to replace source area. The source area can
            be replaced by Poisson sources with the same global rate
            rate_ext ('hom_poisson_stat') or by specific rates
            ('het_poisson_stat') or by time-varying specific current
            ('het_current_nonstat')
        source_area_name: str
            Name of the source area to be replaced.
        cc_input : dict
            Dictionary of cortico-cortical input of the process
            replacing the source area.
        """
        synapses = extract_area_dict(self.network.synapses,
                                     self.network.structure,
                                     self.name,
                                     source_area_name)
        W = extract_area_dict(self.network.W,
                              self.network.structure,
                              self.name,
                              source_area_name)
        v = self.network.params['delay_params']['interarea_speed']
        s = self.network.distances[self.name][source_area_name]
        delay = s / v
        for pop in self.populations:
            for source_pop in self.network.structure[source_area_name]:
                syn_spec = {'weight': W[pop][source_pop],
                            'delay': delay}
                K = synapses[pop][source_pop] / self.neuron_numbers[pop]

                if input_type == 'het_current_nonstat':
                    curr_gen = nest.Create('step_current_generator')
                    dt = self.simulation.params['dt']
                    T = self.simulation.params['t_sim']
                    assert(len(cc_input[source_pop]) == int(T))
                    curr_gen.set({'amplitude_values': K * cc_input[source_pop] * 1e-3,
                                  'amplitude_times': np.arange(dt, T + dt, 1.)})
                    nest.Connect(curr_gen,
                                 self.gids[pop],
                                 syn_spec=syn_spec)
                elif 'poisson_stat' in input_type:  # hom. and het. poisson lead here
                    pg = nest.Create('poisson_generator')
                    pg.set({'rate': K * cc_input[source_pop]})
                    nest.Connect(pg,
                                 self.gids,
                                 syn_spec=syn_spec)


def connect(simulation,
            target_area,
            source_area):
    """
    Connect two areas with each other.

    Parameters
    ----------
    simulation : Simulation instance
        Simulation simulating the network containing the two areas.
    target_area : Area instance
        Target area of the projection
    source_area : Area instance
        Source area of the projection
    """
    network = simulation.network
    synapses = extract_area_dict(network.synapses,
                                 network.structure,
                                 target_area.name,
                                 source_area.name)
    W = extract_area_dict(network.W,
                          network.structure,
                          target_area.name,
                          source_area.name)
    W_sd = extract_area_dict(network.W_sd,
                             network.structure,
                             target_area.name,
                             source_area.name)
    for target in target_area.populations:
        for source in source_area.populations:
            conn_spec = {'rule': 'fixed_total_number',
                         'N': int(synapses[target][source])}

            if target_area == source_area:
                if 'E' in source:
                    w_min = 0.
                    w_max = np.inf
                    mean_delay = network.params['delay_params']['delay_e']
                elif 'I' in source:
                    w_min = -np.inf
                    w_max = 0.
                    mean_delay = network.params['delay_params']['delay_i']
            else:
                w_min = 0.
                w_max = np.inf
                v = network.params['delay_params']['interarea_speed']
                s = network.distances[target_area.name][source_area.name]
                mean_delay = s / v

            syn_spec = {
                'synapse_model': 'static_synapse',
                'weight': nest.math.redraw(
                    nest.random.normal(
                        mean=W[target][source],
                        std=W_sd[target][source]
                        ),
                    min=w_min,
                    max=w_max
                    ),
                'delay': nest.math.redraw(
                    nest.random.normal(
                        mean=mean_delay,
                        std=mean_delay * network.params['delay_params']['delay_rel']
                        ),
                    min=simulation.params['dt'],
                    max=np.inf)}

            nest.Connect(source_area.gids[source],
                         target_area.gids[target],
                         conn_spec,
                         syn_spec)
