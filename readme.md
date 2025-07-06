# MIMO Network Simulation

A modular simulation framework for Cell-Free MIMO networks with jammer detection.

## Project Structure

The project is organized into the following modules:

```
mimo_simulation/
├── __init__.py
├── config.py                 # Configuration parameters
├── models/
│   ├── __init__.py
│   ├── channel.py            # Channel modeling
│   ├── network.py            # Network topology and Graph representation
│   ├── mobility.py           # User movement modeling
│   └── jammer.py             # Jammer behavior
├── simulation/
│   ├── __init__.py
│   ├── simulator.py          # Main simulation engine
│   ├── scenarios.py          # Scenario generators
│   └── batch_runner.py       # Parallel simulation runner
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # Visualization tools
│   ├── data_storage.py       # Data saving/loading utilities
│   └── logging_utils.py      # Logging configuration
└── main.py                   # Entry point script
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/alihossary/cell-free-mimo-simulator.git
cd mimo-simulation
pip install -r requirements.txt
```

## Usage

The simulation can be run in various modes:

### Single Simulation

Run a single simulation with or without a jammer:

```bash
python main.py --mode single --beta 0.5 --steps 80 --with-jammer --tau 5
```

### Batch Simulations

Run multiple simulations with the same parameters:

```bash
python main.py --mode batch --beta 0.5 --tau 5 --steps 80 --normal-count 5 --anomaly-count 5 --use-parallel
```

### Parameter Sweep

Run simulations with different values of a parameter:

```bash
python main.py --mode sweep --sweep-type beta --sweep-values 0.0 0.5 1.0 --steps 80
```

### Parameter Combinations

Run simulations with combinations of different parameters:

```bash
python main.py --mode combinations
```

### Visualizations

Create visualizations of the simulations:

```bash
python main.py --mode visualize --vis-type sweep --sweep-type tau --steps 80
```

## Key Features

1. **Modular Design**: Each component (channel modeling, mobility, etc.) is isolated in its own module for easy maintenance.

2. **Flexible Simulation**: Support for different parameters like Rician fading, jammer activity patterns, and observation windows.

3. **Visualization Tools**: Create static plots and animations of the network topology and connections.

4. **Data Storage**: Save simulation results and metrics for later analysis.

5. **Parallel Processing**: Run multiple simulations in parallel to speed up batch processing.

## Configuration

The main configuration parameters are defined in `config.py`. You can modify these values to change the simulation behavior.

## Web Application Development

This modular structure is designed to be easily extended to a web application. Here's how you can integrate it:

1. **API Layer**: Create REST endpoints that invoke the simulation modules
2. **Frontend**: Build a web interface to configure and run simulations
3. **Visualizations**: Use the visualization utilities to create interactive plots for the web

For example, you could create a Flask or FastAPI backend that exposes endpoints for running simulations and retrieving results, while the frontend could use libraries like D3.js or Plotly to visualize the results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
