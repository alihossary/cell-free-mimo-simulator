# simulation/simulator.py
"""
Main simulation engine for MIMO network.
Coordinates the simulation process and integrates all models.
"""
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional

from models.models_channel import channel_model
from models.models_network import network_model
from models.models_mobility import mobility_model
from models.models_jammer import jammer_model
import config


class MIMOSimulator:
    """Main MIMO network simulator class."""
    
    def __init__(self, n_users: int = config.N_USERS, n_aps: int = config.N_APS,
                n_jammers: int = config.N_JAMMERS, n_j: int = config.N_J,
                n_u: int = config.N_U, n_a: int = config.N_A, 
                area_size: float = config.AREA_SIZE):
        """
        Initialize the MIMO simulator.
        
        Args:
            n_users: Number of users
            n_aps: Number of access points
            n_jammers: Number of jammers
            n_j: Number of antennas per jammer
            n_u: Number of antennas per user
            n_a: Number of antennas per AP
            area_size: Simulation area size (km)
        """
        self.n_users = n_users
        self.n_aps = n_aps
        self.n_jammers = n_jammers
        self.n_j = n_j
        self.n_u = n_u
        self.n_a = n_a
        self.area_size = area_size
        self.ap_positions = config.AP_POSITIONS
    
    def run_normal_simulation(self, beta: float, n_steps: int) -> Tuple[List[nx.Graph], List[np.ndarray]]:
        """
        Run a normal simulation without any jammer.
        
        Args:
            beta: Rician fading parameter
            n_steps: Number of simulation steps
            
        Returns:
            Tuple of (list of network graphs, list of user positions)
        """
        # Initialize user positions and speeds
        user_positions = mobility_model.initialize_positions(self.n_users)
        user_speeds = mobility_model.initialize_speeds(self.n_users)
        
        # Storage for simulation results
        dynamic_graphs = []
        user_positions_history = []
        
        # Run simulation for n_steps
        for step in range(n_steps):
            # Create network graph
            G = network_model.create_network_graph(self.ap_positions, user_positions)
            
            # Calculate channel matrices with Rician fading
            H_rician = channel_model.rician_fading_channel(
                self.n_users, self.n_aps, self.n_u, self.n_a, beta
            )
            H = channel_model.apply_distance_to_channel(
                H_rician, user_positions, self.ap_positions
            )
            
            # Create empty jammer channel matrix
            G_jammer = np.zeros((self.n_users, self.n_j, 1), dtype=complex)
            
            # Assign users to APs
            assignments = network_model.assign_users_to_aps(user_positions, self.ap_positions)
            
            # Update network edges
            network_model.update_network_edges(
                G, self.ap_positions, user_positions, H, G_jammer, 
                assignments, jammer_position=None, jammer_active=False
            )
            
            # Store the graph and positions
            dynamic_graphs.append(G)
            user_positions_history.append(user_positions.copy())
            
            # Update positions for next step
            user_positions = mobility_model.update_positions(
                user_positions, user_speeds, self.ap_positions
            )
            user_positions, user_speeds = mobility_model.reset_boundary_users(
                user_positions, user_speeds
            )
        
        return dynamic_graphs, user_positions_history
    
    def run_jammer_simulation(self, beta: float, tau: int, n_steps: int, 
                            n_frames: int) -> Tuple[List[nx.Graph], bool, List[bool], np.ndarray, List[np.ndarray]]:
        """
        Run a simulation with a jammer.
        
        Args:
            beta: Rician fading parameter
            tau: Jammer active duration within each frame
            n_steps: Number of simulation steps
            n_frames: Number of frames in the simulation
            
        Returns:
            Tuple of (list of network graphs, success flag, jammer states, jammer position, list of user positions)
        """
        max_attempts = 100  # Max attempts to find valid scenario
        
        for attempt in range(max_attempts):
            # Initialize positions and speeds
            user_positions = mobility_model.initialize_positions(self.n_users)
            user_speeds = mobility_model.initialize_speeds(self.n_users)
            jammer_position = jammer_model.initialize_jammer_position()
            
            # Pre-simulate movement to validate scenario
            positions_timeline = []
            current_positions = user_positions.copy()
            
            for step in range(n_steps):
                positions_timeline.append(current_positions.copy())
                current_positions = mobility_model.update_positions(
                    current_positions, user_speeds, self.ap_positions
                )
                current_positions, user_speeds = mobility_model.reset_boundary_users(
                    current_positions, user_speeds
                )
            
            # Check if at least one user enters jammer range
            valid_scenario = jammer_model.is_valid_jammer_scenario(
                positions_timeline, jammer_position
            )
            
            if valid_scenario:
                # Generate jammer activity pattern
                jammer_states = jammer_model.simulate_jammer_activity(n_steps, n_frames, tau)
                
                # Generate graph snapshots
                dynamic_graphs = []
                user_positions_history = []
                current_positions = user_positions.copy()
                current_speeds = user_speeds.copy()
                
                for step in range(n_steps):
                    # Create network graph
                    G = network_model.create_network_graph(
                        self.ap_positions, current_positions, 
                        jammer_position, jammer_states[step]
                    )
                    
                    # Calculate channel matrices with Rician fading
                    H_rician = channel_model.rician_fading_channel(
                        self.n_users, self.n_aps, self.n_u, self.n_a, beta
                    )
                    H = channel_model.apply_distance_to_channel(
                        H_rician, current_positions, self.ap_positions
                    )
                    
                    # Calculate jammer channel matrices
                    G_jammer_rician = channel_model.rician_jammer_channel(
                        self.n_users, self.n_j, beta
                    )
                    G_jammer = channel_model.apply_distance_to_jammer_channel(
                        G_jammer_rician, current_positions, jammer_position
                    )
                    
                    # Assign users to APs
                    assignments = network_model.assign_users_to_aps(
                        current_positions, self.ap_positions
                    )
                    
                    # Update network edges
                    network_model.update_network_edges(
                        G, self.ap_positions, current_positions, H, G_jammer,
                        assignments, jammer_position, jammer_states[step]
                    )
                    
                    # Store results
                    dynamic_graphs.append(G)
                    user_positions_history.append(current_positions.copy())
                    
                    # Update positions for next step
                    current_positions = mobility_model.update_positions(
                        current_positions, current_speeds, self.ap_positions
                    )
                    current_positions, current_speeds = mobility_model.reset_boundary_users(
                        current_positions, current_speeds
                    )
                
                return dynamic_graphs, True, jammer_states, jammer_position, user_positions_history
        
        # If no valid scenario found after max attempts
        return None, False, None, None, None

# Create a singleton instance for global use
mimo_simulator = MIMOSimulator()
