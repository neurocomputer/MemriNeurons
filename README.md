# Welcome to MemriNeurons — a Framework for In-Memory Computing and Neuromorphic Systems Based on Memristors

The main documentation for environment setup is available here: [github.com/neurocomputer/MemriDocs](https://github.com/neurocomputer/MemriDocs)

## ⚙️ Quick Environment Setup

To run MemriNeurons, you need to:

1. First clone the **MemriBoard** repository.
2. Then clone the **MemriNeurons** repository inside it.
3. If you are working with a real board (not the simulator), you also need to clone the **MemriCore** repository.

Neural networks must be created inside the `MemriBoard` folder:

```
MemriBoard
        |-MemriNeurons
        |-MemriCORE
        |-simulator.cb
        |-your_neural_network_file.py
```

Example scripts can be found in the `examples` folder of <https://github.com/neurocomputer/MemriDocs>.
The neural network creation pipeline is described in [`docs/en_memrineurons.md`](docs/en_memrineurons.md).

## 📁 MemriNeurons Structure

### `components.py`
Contains neural network components for conversion, so that a model can be run using only NumPy:

- `load_model` — load a model
- `feature_map_reshape` — feature map reshape helper for processing in a convolutional layer
- `Sequential` — neural network model
- `Layer` — neural network layer
- `Dense` — fully connected layer
- `Conv2D` — convolutional layer
- `Flatten` — flatten layer

### `keras2nmp.py`
Converter for neural network models from Keras format to NumPy format:

- `convert_keras_2_nmp` — conversion function

### `hardlayers.py`
Contains classes that perform matrix multiplication in neural network layers:

- `ElementWiseMatMulLayer` — element-wise multiplication for an entire layer
- other layers are under development — join us!

### `cores.py`
Compute engine on a memristive crossbar array. There is a board containing memristors. A driver from **MemriCore** is used to interact with the board, but it only handles data transmission over the interface (in modes `mode_7`, `mode_9`, `mode_mvm`, `mode_core`). The core, however, also needs additional functions such as reading/writing weights, etc.:

- `HardCore` — class for working with the core (description here: [`docs/en_hardcore.md`](docs/en_hardcore.md))

### `snn.py`
Contains components of spiking neural networks:

- `Astrocyte` — astrocytes
- `Synapse` — synapses
- `LIFNeuron` — neurons

### `src.py`
Contains auxiliary functions:

- `get_logger` — get a logger
- `change_log_file` — change the log file (uses `get_file_handler`)
- `poisson_binary_array` — create an array of zeros and ones following a Poisson distribution

---

## 🚀 Let's Get Started!