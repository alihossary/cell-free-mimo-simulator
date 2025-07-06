# models/network.py
"""
Network topology model for MIMO simulation.
Includes graph representation, user-AP assignments, and SINR calculations.
"""
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List

class NetworkModel:
    """Model for network topology, AP assignments, and signal calculations."""
    
    def __init__(self, noise_power: float = 0.0002, gamma_0: float = 4, jammer_range: float = 0.35,
                jammer_power: float = 1.2):
        """
        Initialize the network model.
        
        Args:
            noise_power: AWGN noise power
            gamma_0: Minimum SINR threshold (dB)
            jammer_range: Maximum range of jammer effect (km)
            jammer_power: Jammer transmission power
        """
        self.noise_power = noise_power
        self.gamma_0 = gamma_0
        self.jammer_range = jammer_range
        self.jammer_power = jammer_power
    
    def create_network_graph(self, ap_positions: np.ndarray, user_positions: np.ndarray, 
                           jammer_position: Optional[np.ndarray] = None, 
                           jammer_active: bool = False) -> nx.Graph:
        """
        Create a network graph with nodes for APs, users, and optionally jammer.
        
        Args:
            ap_positions: Positions of access points
            user_positions: Positions of users
            jammer_position: Position of jammer (if present)
            jammer_active: Whether jammer is active
            
        Returns:
            NetworkX graph representing the network
        """
        G = nx.Graph()
        
        # Add AP nodes
        for i, pos in enumerate(ap_positions):
            G.add_node(f'AP{i+1}', pos=pos, type='AP')
        
        # Add user nodes
        for i, pos in enumerate(user_positions):
            G.add_node(f'User{i+1}', pos=pos, type='User')
        
        # Add jammer node if present
        if jammer_position is not None:
            G.add_node('Jammer', pos=jammer_position, type='Jammer', active=jammer_active)
        
        return G
    
    def assign_users_to_aps(self, user_positions: np.ndarray, 
                          ap_positions: np.ndarray) -> Dict[int, int]:
        """
        Assign each user to the closest AP using an iterative approach.
        
        Args:
            user_positions: Positions of users
            ap_positions: Positions of access points
            
        Returns:
            Dictionary mapping user indices to their assigned AP indices
        """
        assignments = {}
        
        # Make copies of user and AP lists to remove from
        available_users = list(range(len(user_positions)))
        available_aps = list(range(len(ap_positions)))
        
        while available_users and available_aps:
            min_distance = float('inf')
            best_pair = None
            
            # Find the user-AP pair with minimum distance
            for user_idx in available_users:
                for ap_idx in available_aps:
                    distance = np.linalg.norm(user_positions[user_idx] - ap_positions[ap_idx])
                    if distance < min_distance:
                        min_distance = distance
                        best_pair = (user_idx, ap_idx)
            
            if best_pair:
                user_idx, ap_idx = best_pair
                # Assign this user to this AP
                assignments[user_idx] = ap_idx
                
                # Remove from available lists
                available_users.remove(user_idx)
                available_aps.remove(ap_idx)
            else:
                break
        
        return assignments
    
    def calculate_sinr(self, user_idx: int, assigned_ap_idx: int, 
                  user_positions: np.ndarray, ap_positions: np.ndarray, 
                  h: np.ndarray, s: np.ndarray,
                  jammer_position: Optional[np.ndarray] = None,
                  jammer_active: bool = True) -> Tuple[float, float, float, float]:
        """
        Calculate SINR with Maximal Ratio Combining (MRC) accounting for spatial
        correlation and jammer effects.
        """
        # Desired channel vector
        h_km = h[user_idx, assigned_ap_idx]  # Shape: [N_U, N_A]
        desired_power = np.linalg.norm(h_km)**2

        # 2. Calculate interference with spatial correlation
        interference = 0
        for other_ap_idx in range(len(ap_positions)):
            if other_ap_idx != assigned_ap_idx:
                h_km_prime = h[user_idx, other_ap_idx]
                h_km_norm_squared = np.linalg.norm(h_km)**2
                if h_km_norm_squared > 0:
                    correlation = np.abs(np.vdot(h_km.flatten(), h_km_prime.flatten()))**2 / h_km_norm_squared
                    interference += correlation * np.linalg.norm(h_km_prime)**2

        # 3. Jammer interference
        jammer_interference = 0
        if jammer_position is not None and jammer_active:
            jammer_distance = np.linalg.norm(user_positions[user_idx] - jammer_position)
            if jammer_distance < self.jammer_range:
                s_k = s[user_idx].flatten()  # shape: (N_J,)
                jammer_interference = self.jammer_power * np.linalg.norm(s_k)**2

        # 4. Calculate SINR
        sinr = desired_power / (self.noise_power + interference + jammer_interference)
        sinr_db = 10 * np.log10(sinr)
        return sinr_db, desired_power, interference, jammer_interference
    
    def update_network_edges(self, G: nx.Graph, ap_positions: np.ndarray, 
                           user_positions: np.ndarray, h: np.ndarray, 
                           s: np.ndarray, assignments: Dict[int, int], 
                           jammer_position: Optional[np.ndarray] = None,
                           jammer_active: bool = False) -> None:
        """
        Update graph edges based on SINR calculations and user-AP assignments.
        
        Args:
            G: NetworkX graph to update
            ap_positions: Positions of access points
            user_positions: Positions of users
            h: Channel matrices
            s: Jammer channel matrices
            assignments: User-AP assignments dictionary
            jammer_position: Position of jammer (if any)
            jammer_active: Whether jammer is active
        """
        # Clear existing edges
        G.remove_edges_from(list(G.edges()))
        
        # For each user, check if SINR meets threshold for its assigned AP
        for user_idx, assigned_ap_idx in assignments.items():
            # Calculate SINR with full interference model
            sinr_db, desired_power, interference, jammer_interference = self.calculate_sinr(
                user_idx, assigned_ap_idx, user_positions, ap_positions,
                h, s, jammer_position, jammer_active
            )
            
            # Connection condition: Γ(k,m) > Γ_0
            if sinr_db > self.gamma_0:
                # Calculate the distance for edge weight
                distance = np.linalg.norm(user_positions[user_idx] - ap_positions[assigned_ap_idx])
                
                # Add edge with various attributes for analysis
                G.add_edge(
                    f'AP{assigned_ap_idx+1}',
                    f'User{user_idx+1}',
                    sinr=sinr_db,
                    distance=distance,
                    desired_gain=desired_power,
                    interference=interference,
                    jammer_interference=jammer_interference
                )

# Generate a singleton instance for global use
network_model = NetworkModel()
