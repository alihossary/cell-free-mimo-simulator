# Project structure overview

# mimo_simulation/
# ├── __init__.py
# ├── config.py                 # Configuration parameters
# ├── models/
# │   ├── __init__.py
# │   ├── channel.py            # Channel modeling
# │   ├── network.py            # Network topology and Graph representation
# │   ├── mobility.py           # User movement modeling
# │   └── jammer.py             # Jammer behavior
# ├── simulation/
# │   ├── __init__.py
# │   ├── simulator.py          # Main simulation engine
# │   ├── scenarios.py          # Scenario generators
# │   └── batch_runner.py       # Parallel simulation runner
# ├── utils/
# │   ├── __init__.py
# │   ├── visualization.py      # Visualization tools
# │   ├── data_storage.py       # Data saving/loading utilities
# │   └── logging_utils.py      # Logging configuration
# └── main.py                   # Entry point script

# config.py
"""
Configuration parameters for MIMO network simulation.
"""
import numpy as np
import os

# System parameters
RAMDISK_BASE = '/dev/shm/sim_temp'
os.makedirs(RAMDISK_BASE, exist_ok=True)

# Network parameters
N_APS = 5                # Number of Access Points
N_USERS = 10             # Number of Users
N_JAMMERS = 1            # Number of Jammers
N_J = 4                  # Number of antennas per jammer
N_U = 1                  # Number of antennas per user
N_A = 4                  # Number of antennas per AP
AREA_SIZE = 1.0          # 1 km x 1 km area

# Channel parameters
NOISE_POWER = 0.0002     # AWGN noise power (sigma^2)
GAMMA_0 = 4              # Minimum SINR threshold (dB)
D_0 = 0.1                # Reference distance for path loss (km)

# Movement parameters
SPEED_MAX = 0.001667     # Maximum speed (km/s) [6 km/h ≈ 0.001667 km/s]
T = 1                    # Time step (seconds)
DISTANCE_THRESHOLD = 0.4 # Max distance for a connection (km)

# Jammer parameters
JAMMER_RANGE = 0.35      # Jammer impact distance (km)
JAMMER_POWER = 1.2       # Default jammer power
INTERFERENCE_RANGE = 0.7 # Interference range (km)

# Default simulation values
DEFAULT_BETA_VALUES = [0, 0.5, 1]  # Rician fading parameter values
DEFAULT_TAU_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Jammer activity durations
DEFAULT_NSTEPS_VALUES = [40, 60, 80, 100, 120]  # Simulation step counts
DEFAULT_NORMAL_COUNT = 2  # Number of normal simulations
DEFAULT_ANOMALY_COUNT = 2  # Number of anomaly simulations

# Initialize AP positions
AP_POSITIONS = np.array([
    [0.2, 0.2],  # Bottom-left
    [0.8, 0.2],  # Bottom-right
    [0.2, 0.8],  # Top-left
    [0.8, 0.8],  # Top-right
    [0.5, 0.5]   # Center
])
