# benchmark-models

A collection of models adapted to interface with the [nest benchmarking framework](https://github.com/INM-6/nest_benchmarking_framework). Here, documentation on how to interface a new model is provided (README, section _Developing the framework_).



### Input

In order to include a model to the JUBE workflow it needs to be able to receive input from the JUBE benchmark file. The easiest way is to use placeholders in the simulation script which are substituted with values defined in the corresponding JUBE file. For example the hpc\_benchmark.yaml contains this substituteset:

```
substituteset:
      name: model_substitutions
      iofile: {in: hpc_benchmark.py, out: hpc_benchmark.py}
      sub:
      - {source: "{num_vps}", dest: $num_vps}
      - {source: "{record_spikes}", dest: $record_spikes}
      - {source: "{model_time_sim}", dest: $model_time_sim}
      - {source: "{model_time_presim}", dest: $model_time_presim}
      - {source: "{N_SCALING}", dest: $scale}
      - {source: "{rng_seed}", dest: $rng_seed}
```

This substitueset is catched like this in the hpc\_benchmark.py simulation file:

```
params = {
    'nvp': {num_vps},                  # total number of virtual processes
    'scale': {N_SCALING},              # scaling factor of the network size
                                       # total network size = scale*11250 neurons
    'simtime': {model_time_sim},       # total simulation time in ms
    'presimtime': {model_time_presim}, # simulation time until reaching equilibrium
    'dt': 0.1,                         # simulation step
    'record_spikes': {record_spikes},  # switch to record spikes of excitatory
                                       # neurons to file
    'rng_seed': {rng_seed},            # random number generator seed
    'path_name': '.',                  # path where all files will have to be written
    'log_file': 'logfile',             # naming scheme for the log files
}
```

Depending on the usecase and model specifications this substituteset can be extended, as done for example in the microcircuit. A model might require a different way of providing the input, see for example the multi-area model benchmark where the input is passed via command line arguments.

### Placement of Python level timers

### Output
