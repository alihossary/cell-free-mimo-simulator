# models/mobility.py
"""
Mobility model for MIMO network simulation.
Handles user movement patterns and position updates.
"""
import numpy as np
from typing import Tuple

class MobilityModel:
    """Model for user mobility patterns and position updates."""
    
    def __init__(self, area_size: float = 1.0, speed_max: float = 0.001667, 
                time_step: float = 1.0, noise_std: float = 0.0008):
        """
        Initialize the mobility model.
        
        Args:
            area_size: Size of the simulation area (km)
            speed_max: Maximum speed of users (km/s)
            time_step: Simulation time step (s)
            noise_std: Standard deviation of mobility noise
        """
        self.area_size = area_size
        self.speed_max = speed_max
        self.time_step = time_step
        self.noise_std = noise_std
    
    def initialize_positions(self, n_users: int) -> np.ndarray:
        """
        Initialize random user positions within the area.
        
        Args:
            n_users: Number of users to initialize
            
        Returns:
            Array of user positions
        """
        return np.random.uniform(0, self.area_size, (n_users, 2))
    
    def initialize_speeds(self, n_users: int) -> np.ndarray:
        """
        Initialize random user speeds.
        
        Args:
            n_users: Number of users
            
        Returns:
            Array of user speeds
        """
        return np.random.uniform(-self.speed_max, self.speed_max, (n_users, 2))
    
    def update_positions(self, user_positions: np.ndarray, user_speeds: np.ndarray, 
                        ap_positions: np.ndarray) -> np.ndarray:
        """
        Update user positions based on the mobility model.
        
        Args:
            user_positions: Current user positions
            user_speeds: Current user speeds
            ap_positions: Access point positions (for collision avoidance)
            
        Returns:
            Updated user positions
        """
        # Generate random Gaussian noise
        random_noise = np.random.normal(0, self.noise_std, (len(user_positions), 2))
        
        # Update user positions: [x,y](n+1) = [x,y](n) + (v + w) * T
        updated_positions = user_positions + (user_speeds + random_noise) * self.time_step
        
        # Check distance to APs and reset if too close
        min_distance = 0.02  # 20m minimum distance
        for i, user_pos in enumerate(updated_positions):
            for ap_pos in ap_positions:
                vector_to_ap = ap_pos - user_pos
                distance_to_ap = np.linalg.norm(vector_to_ap)
                if distance_to_ap < min_distance:
                    # Calculate the vector to push the user away from the AP
                    push_vector = vector_to_ap / distance_to_ap  # Normalize
                    push_distance = min_distance - distance_to_ap
                    updated_positions[i] = user_pos - push_vector * push_distance
        
        return updated_positions
    
    def reset_boundary_users(self, user_positions: np.ndarray, user_speeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset users that hit the boundary and update speeds.
        
        Args:
            user_positions: Current user positions
            user_speeds: Current user speeds
            
        Returns:
            Tuple of (updated_positions, updated_speeds)
        """
        updated_positions = user_positions.copy()
        updated_speeds = user_speeds.copy()
        
        for i, pos in enumerate(user_positions):
            # Reset position if outside boundary
            if (pos[0] < 0 or pos[0] > self.area_size or 
                pos[1] < 0 or pos[1] > self.area_size):
                # Reset to random position within the area
                updated_positions[i] = np.random.uniform(0, self.area_size, 2)
                # Assign a new random velocity
                updated_speeds[i] = np.random.uniform(-self.speed_max, self.speed_max, 2)
            
            # Bounce off walls (alternative to resetting)
            else:
                if pos[0] <= 0 or pos[0] >= self.area_size:
                    updated_speeds[i, 0] *= -1  # Reverse x direction
                if pos[1] <= 0 or pos[1] >= self.area_size:
                    updated_speeds[i, 1] *= -1  # Reverse y direction
        
        return updated_positions, updated_speeds

# Generate a singleton instance for global use
mobility_model = MobilityModel()
