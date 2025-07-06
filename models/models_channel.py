# models/channel.py
"""
Channel modeling for MIMO network simulation.
Includes Rician fading channel models and distance-based path loss.
"""
import numpy as np
from typing import Optional

class ChannelModel:
    """Model for generating and manipulating wireless channels."""
    
    def __init__(self, d_0: float = 0.1):
        """
        Initialize the channel model.
        
        Args:
            d_0 (float): Reference distance for path loss model (km).
        """
        self.d_0 = d_0
    
    def rician_fading_channel(self, n_users: int, n_aps: int, n_u: int, n_a: int, beta: float) -> np.ndarray:
        """
        Generate a Rician fading channel matrix h(k,m,n) with parameter beta.
        beta=0 gives deterministic model, beta=1 gives fully random model.
        
        Args:
            n_users: Number of users
            n_aps: Number of access points
            n_u: Number of antennas per user
            n_a: Number of antennas per AP
            beta: Rician fading parameter (0=deterministic, 1=random)
            
        Returns:
            Channel matrices without distance factor applied yet.
        """
        # Stochastic component (g(k,m,n) in the formula)
        stochastic_component = np.random.normal(0, 1/np.sqrt(2), (n_users, n_aps, n_u, n_a)) + \
                              1j * np.random.normal(0, 1/np.sqrt(2), (n_users, n_aps, n_u, n_a))
        
        # Deterministic component (the sigma_{k,m}(n) term)
        deterministic_component = np.ones((n_users, n_aps, n_u, n_a), dtype=complex)
        
        # Formula: h(k,m,n) = β * sigma_{k,m}(n) + sqrt(1-β²) * g(k,m,n)
        channel_matrix = beta * deterministic_component + np.sqrt(1-beta**2) * stochastic_component
        
        return channel_matrix
    
    def apply_distance_to_channel(self, h: np.ndarray, user_positions: np.ndarray, 
                                ap_positions: np.ndarray) -> np.ndarray:
        """
        Apply distance-based path loss to channel matrices.
        
        Args:
            h: Channel matrix
            user_positions: Positions of users
            ap_positions: Positions of access points
            
        Returns:
            Channel matrix with path loss applied
        """
        n_users, n_aps = user_positions.shape[0], ap_positions.shape[0]
        channel_with_distance = np.zeros_like(h, dtype=complex)
        
        for k in range(n_users):
            for m in range(n_aps):
                # Calculate distance between user k and AP m
                distance = np.linalg.norm(user_positions[k] - ap_positions[m])
                if distance > 0:
                    # Path loss factor: (d_0)² / (d_{k,m}(n))²
                    path_loss_factor = np.sqrt((self.d_0**2) / (distance**2))
                    # Apply to the channel matrix
                    channel_with_distance[k, m] = h[k, m] * path_loss_factor
                else:
                    # Avoid division by zero
                    channel_with_distance[k, m] = h[k, m]
                    
        return channel_with_distance
    
    def rician_jammer_channel(self, n_users: int, n_j: int, beta: float) -> np.ndarray:
        """
        Generate Rician fading channel matrix from jammer to users.
        
        Args:
            n_users: Number of users
            n_j: Number of jammer antennas
            beta: Rician fading parameter
            
        Returns:
            Jammer-to-user channel matrix
        """
        stochastic_component = np.random.normal(0, 1/np.sqrt(2), (n_users, n_j, 1)) + \
                              1j * np.random.normal(0, 1/np.sqrt(2), (n_users, n_j, 1))
        
        deterministic_component = np.ones((n_users, n_j, 1), dtype=complex)
        
        return beta * deterministic_component + np.sqrt(1-beta**2) * stochastic_component
    
    def apply_distance_to_jammer_channel(self, s: np.ndarray, user_positions: np.ndarray, 
                                       jammer_position: Optional[np.ndarray]) -> np.ndarray:
        """
        Apply distance-based path loss to jammer channel.
        
        Args:
            s: Jammer channel matrix
            user_positions: User positions
            jammer_position: Jammer position (or None if no jammer)
            
        Returns:
            Jammer channel with path loss applied
        """
        if jammer_position is None:
            return np.zeros_like(s, dtype=complex)
        
        n_users = user_positions.shape[0]
        s_scaled = np.zeros_like(s, dtype=complex)
        
        for k in range(n_users):
            distance = np.linalg.norm(user_positions[k] - jammer_position)
            if distance > 0:
                scale = np.sqrt((self.d_0 ** 2) / (distance ** 2))
                s_scaled[k] = s[k] * scale
            else:
                s_scaled[k] = s[k]
                
        return s_scaled

# Generate a singleton instance for global use
channel_model = ChannelModel()
