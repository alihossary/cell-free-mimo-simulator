# models/jammer.py
"""
Jammer model for MIMO network simulation.
Handles jammer position and activity patterns.
"""
import numpy as np
from typing import List, Optional, Tuple

class JammerModel:
    """Model for jammer behavior and activity patterns."""
    
    def __init__(self, area_size: float = 1.0, jammer_range: float = 0.35):
        """
        Initialize the jammer model.
        
        Args:
            area_size: Size of the simulation area (km)
            jammer_range: Maximum range of jammer effect (km)
        """
        self.area_size = area_size
        self.jammer_range = jammer_range
    
    def initialize_jammer_position(self) -> np.ndarray:
        """
        Initialize a random jammer position within the area.
        
        Returns:
            Jammer position as a 2D numpy array
        """
        return np.random.uniform(0, self.area_size, 2)
    
    def simulate_jammer_activity(self, n_steps: int, n_frames: int, tau: int) -> List[bool]:
        """
        Simulate jammer activity pattern based on frames and active duration.
        
        Args:
            n_steps: Total simulation steps
            n_frames: Number of frames in total observation window
            tau: Active duration within each frame [0, frame_length]
            
        Returns:
            List of boolean jammer states for each step
        """
        steps_per_frame = n_steps // n_frames
        
        # Calculate active steps per frame (proportion of tau to frame duration)
        active_steps_per_frame = int(steps_per_frame * (tau / steps_per_frame))
        
        # Create pattern
        jammer_pattern = []
        for frame in range(n_frames):
            # Active for the first 'active_steps_per_frame' steps of each frame
            for step in range(steps_per_frame):
                jammer_pattern.append(step < active_steps_per_frame)
        
        # Ensure the pattern is the right length
        return jammer_pattern[:n_steps]
    
    def is_valid_jammer_scenario(self, user_positions_timeline: List[np.ndarray], 
                               jammer_position: np.ndarray) -> bool:
        """
        Check if any user enters jammer range during simulation.
        
        Args:
            user_positions_timeline: List of user positions at each time step
            jammer_position: Position of the jammer
            
        Returns:
            True if at least one user enters jammer range, False otherwise
        """
        for positions in user_positions_timeline:
            for user_pos in positions:
                if np.linalg.norm(user_pos - jammer_position) < self.jammer_range:
                    return True
        return False
    
    def find_valid_jammer_position(self, user_positions_timeline: List[np.ndarray], 
                                 max_attempts: int = 100) -> Optional[np.ndarray]:
        """
        Find a valid jammer position where at least one user enters jammer range.
        
        Args:
            user_positions_timeline: List of user positions at each time step
            max_attempts: Maximum number of attempts to find a valid position
            
        Returns:
            Valid jammer position or None if not found after max attempts
        """
        for _ in range(max_attempts):
            jammer_position = self.initialize_jammer_position()
            if self.is_valid_jammer_scenario(user_positions_timeline, jammer_position):
                return jammer_position
        return None

# Generate a singleton instance for global use
jammer_model = JammerModel()
