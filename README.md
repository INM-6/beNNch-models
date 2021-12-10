# beNNch-models

A collection of models adapted to interface with [beNNch](https://github.com/INM-6/beNNch). An overview on how to interface a new model can be found [here](https://github.com/INM-6/beNNch#developer-guide). A more detailed description is provided below.

### Input

In order to include a model to the JUBE workflow it needs to be able to receive input from its JUBE benchmark file. We recommend using placeholders in the simulation script which can be substituted by JUBE with values defined in the corresponding JUBE config file. For example, the [simulation file of the HPC-benchmark](./hpc_benchmark/hpc_benchmark.py) contains the following placeholders:

```python
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

The corresponding `subsituteset` of the [JUBE benchmarking file](https://github.com/INM-6/beNNch/blob/main/benchmarks/hpc_benchmark_31.yaml) then looks as follows:

```yaml
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
where `$<parameter>` are parameters defined in either the model-specific config file or in the JUBE benchmarking script.

Depending on the use case and model specifications this `substituteset` can be extended, as done for example in the microcircuit. A model might require a different way of providing the input, see for example the multi-area model benchmark where the input is passed via command line arguments.

### Python--level measurements

In the provided models, several blocks of code are timed on the python level via wrappings of `time.time()` from the `time` library. These timers are common:

* py\_time\_create
    * Time for creation of nodes (neuronal populations, devices) and setting of  initial membrane potentials.
* py\_time\_connect
    * Time for connecting nodes with each other, devices with nodes and preparing the connection infrastructure (i.e., calls to `nest.Prepare()`).
* py\_time\_presimulate
    * Time for the pre-simulation phase. This is defined as the first call to `nest.Simulate(t_presim)` which simulates the network for a short amount of time until it reaches its stable state.
* py\_time\_simulate
    * Time for the actual simulation phase. This timer is a wrapper around `nest.Simulate(t_sim)`

Furthermore, it is useful to record some information on memory consumption.

* base\_memory
    * Memory consumption before any NEST related operations have been executed.
* node\_memory
    * Memory consumption after creation of all nodes.
* network\_memory
    * Memory consumption after connection of all nodes.
* init\_memory
    * Memory consumption after pre-simulation.
* total\_memory
    * Memory consumption after the simulation has finished.

### Output

The timing and memory data mentioned above should be written out to process-private logging files by every process. `beNNch` takes care of combining those to single result files by averaging (or summing) the recorded quantities.
Furthermore, a complete dump of the NEST KernelStatus should be written out to file to keep track of simulation metadata. This data can be accessed via `nest.GetKernelStatus()`.
