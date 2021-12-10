# benchmark-models

A collection of models adapted to interface with the [nest benchmarking framework](https://github.com/INM-6/nest_benchmarking_framework). An overview on how to interface a new model is provided [here](https://github.com/INM-6/beNNch#developer-guide). A more detailed description is provided below.

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

### Python level measurement devices

On the python level several blocks of code are measured. These blocks are timed via wrapping `time.time()` from the time library around these blocks of code. These timers are common:

* py\_time\_create
    * This is the time that the creation of nodes (neuronal populations, devices) and the setting of the initial membrane potentials takes.
* py\_time\_connect
    * This is the time that connecting nodes with each other, devices with nodes and preparing the connection infrastructure (i.e. call to `nest.Prepare()`) take.
* py\_time\_presimulate
    * Time that the presimulation phase takes. This is the first call to `nest.Simulate(t_presim)` which simulates the network for a short amount of time until it reaches its stable state.
* py\_time\_simulate
    * This is the actual simulation phase. This timer is a wrapper around `nest.Simulate(t_sim)`

Furthermore it is useful to write out some information of memory consumption.

* base\_memory
    * Memory consumption before any NEST related operation has been done.
* node\_memory
    * Memory consumption after Creation of all nodes.
* network\_memory
    * Memory consumption after Connection of all nodes.
* init\_memory
    * Memory consumption after Presimulation.
* total\_memory
    * Memory consumption after Simulation has finished.

### Output

The timing and memory data mentioned above should be written out to logging files by every task to task private files. Furthermore a complete dump of the nest Kernel status should be written out to this file. This data can be accessed via `nest.GetKernelStatus()`.
