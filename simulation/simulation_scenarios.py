# simulation/scenarios.py
"""
Scenario generators for MIMO network simulation.
Provides functions for creating different simulation scenarios.
"""
import numpy as np
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

from models.models_channel import channel_model
from models.models_network import network_model
from models.models_mobility import mobility_model
from models.models_jammer import jammer_model
from simulation.simulation_simulator import mimo_simulator
from utils.utils_data_storage import save_simulation_results, extract_simulation_metrics, generate_simulation_summary
from utils.utils_logging import default_logger as logger
import config

def generate_scenario_id(prefix: str = "") -> str:
    """
    Generate a unique scenario ID based on timestamp and UUID.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique scenario ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{uid}" if prefix else f"{timestamp}_{uid}"

def create_normal_scenario(beta: float = 0.0, n_steps: int = 80, 
                         scenario_id: Optional[str] = None) -> str:
    """
    Create and save a normal scenario without jammer.
    
    Args:
        beta: Rician fading parameter
        n_steps: Number of simulation steps
        scenario_id: Optional scenario ID (generated if not provided)
        
    Returns:
        ID of the created scenario
    """
    # Generate scenario ID if not provided
    if scenario_id is None:
        scenario_id = generate_scenario_id("normal")
    
    logger.info(f"Creating normal scenario {scenario_id} with beta={beta}, n_steps={n_steps}")
    
    # Run normal simulation
    graphs, user_positions = mimo_simulator.run_normal_simulation(beta, n_steps)
    
    # Store simulation parameters
    params = {
        "scenario_type": "normal",
        "beta": beta,
        "n_steps": n_steps,
        "n_users": config.N_USERS,
        "n_aps": config.N_APS,
        "n_antennas_user": config.N_U,
        "n_antennas_ap": config.N_A,
        "area_size": config.AREA_SIZE,
        "noise_power": config.NOISE_POWER,
        "gamma_0": config.GAMMA_0,
        "speed_max": config.SPEED_MAX,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save simulation results
    sim_dir = save_simulation_results(
        scenario_id, 
        graphs, 
        params, 
        user_positions=user_positions
    )
    
    # Extract and save metrics
    metrics = extract_simulation_metrics(graphs)
    summary_file = generate_simulation_summary(scenario_id, metrics, params)
    
    logger.info(f"Normal scenario {scenario_id} created successfully")
    logger.info(f"Results saved to {sim_dir}")
    
    return scenario_id

def create_jammer_scenario(beta: float = 0.0, tau: int = 5, n_steps: int = 80, 
                         n_frames: int = 8, scenario_id: Optional[str] = None,
                         max_attempts: int = 100) -> Optional[str]:
    """
    Create and save a scenario with jammer.
    
    Args:
        beta: Rician fading parameter
        tau: Jammer active duration within each frame
        n_steps: Number of simulation steps
        n_frames: Number of frames in the simulation
        scenario_id: Optional scenario ID (generated if not provided)
        max_attempts: Maximum attempts to create a valid scenario
        
    Returns:
        ID of the created scenario or None if failed
    """
    # Generate scenario ID if not provided
    if scenario_id is None:
        scenario_id = generate_scenario_id("jammer")
    
    logger.info(f"Creating jammer scenario {scenario_id} with beta={beta}, tau={tau}, n_steps={n_steps}")
    
    # Try to create a valid jammer scenario
    for attempt in range(max_attempts):
        # Run jammer simulation
        graphs, success, jammer_states, jammer_position, user_positions = (
            mimo_simulator.run_jammer_simulation(beta, tau, n_steps, n_frames)
        )
        
        if success:
            # Store simulation parameters
            params = {
                "scenario_type": "jammer",
                "beta": beta,
                "tau": tau,
                "n_steps": n_steps,
                "n_frames": n_frames,
                "n_users": config.N_USERS,
                "n_aps": config.N_APS,
                "n_jammers": config.N_JAMMERS,
                "n_antennas_user": config.N_U,
                "n_antennas_ap": config.N_A,
                "n_antennas_jammer": config.N_J,
                "area_size": config.AREA_SIZE,
                "noise_power": config.NOISE_POWER,
                "gamma_0": config.GAMMA_0,
                "jammer_range": config.JAMMER_RANGE,
                "jammer_power": config.JAMMER_POWER,
                "speed_max": config.SPEED_MAX,
                "timestamp": datetime.now().isoformat(),
                "attempt": attempt + 1
            }
            
            # Save simulation results
            sim_dir = save_simulation_results(
                scenario_id, 
                graphs, 
                params, 
                user_positions=user_positions,
                jammer_states=jammer_states,
                jammer_position=jammer_position
            )
            
            # Extract and save metrics
            metrics = extract_simulation_metrics(graphs)
            summary_file = generate_simulation_summary(scenario_id, metrics, params)
            
            logger.info(f"Jammer scenario {scenario_id} created successfully on attempt {attempt+1}")
            logger.info(f"Results saved to {sim_dir}")
            
            return scenario_id
        
        logger.warning(f"Failed to create valid jammer scenario on attempt {attempt+1}/{max_attempts}")
    
    logger.error(f"Failed to create valid jammer scenario after {max_attempts} attempts")
    return None

def create_parameter_sweep_scenarios(
    sweep_type: str = 'beta', 
    values: Optional[List[float]] = None,
    n_steps: int = 80,
    scenario_count: int = 3
) -> Dict[str, List[str]]:
    """
    Create multiple scenarios with different parameter values.
    
    Args:
        sweep_type: Parameter to sweep ('beta', 'tau', or 'n_steps')
        values: List of parameter values to sweep
        n_steps: Number of simulation steps (if not sweeping n_steps)
        scenario_count: Number of scenarios to generate for each parameter value
        
    Returns:
        Dictionary mapping parameter values to lists of scenario IDs
    """
    # Set default values if not provided
    if values is None:
        if sweep_type == 'beta':
            values = [0.0, 0.5, 1.0]
        elif sweep_type == 'tau':
            values = [1, 3, 5, 7, 10]
        elif sweep_type == 'n_steps':
            values = [40, 60, 80, 100, 120]
        else:
            raise ValueError(f"Unknown sweep_type: {sweep_type}")
    
    # Create scenarios for each value
    scenarios = {}
    
    for value in values:
        scenarios[value] = []
        
        for i in range(scenario_count):
            if sweep_type == 'beta':
                # Create both normal and jammer scenarios
                normal_id = create_normal_scenario(
                    beta=value, 
                    n_steps=n_steps, 
                    scenario_id=f"normal_beta{value}_{i}"
                )
                scenarios[value].append(normal_id)
                
                jammer_id = create_jammer_scenario(
                    beta=value, 
                    n_steps=n_steps, 
                    scenario_id=f"jammer_beta{value}_{i}"
                )
                if jammer_id:
                    scenarios[value].append(jammer_id)
                
            elif sweep_type == 'tau':
                # Create only jammer scenarios (tau doesn't affect normal scenarios)
                jammer_id = create_jammer_scenario(
                    tau=int(value), 
                    n_steps=n_steps, 
                    scenario_id=f"jammer_tau{value}_{i}"
                )
                if jammer_id:
                    scenarios[value].append(jammer_id)
                
            elif sweep_type == 'n_steps':
                # Create both normal and jammer scenarios
                normal_id = create_normal_scenario(
                    n_steps=int(value), 
                    scenario_id=f"normal_steps{value}_{i}"
                )
                scenarios[value].append(normal_id)
                
                jammer_id = create_jammer_scenario(
                    n_steps=int(value), 
                    scenario_id=f"jammer_steps{value}_{i}"
                )
                if jammer_id:
                    scenarios[value].append(jammer_id)
    
    return scenarios

def create_combination_scenarios(
    beta_values: List[float] = [0.0, 1.0],
    tau_values: List[int] = [2, 5, 8],
    n_steps_values: List[int] = [80],
    scenario_count: int = 1
) -> Dict[str, List[str]]:
    """
    Create scenarios with combinations of different parameters.
    
    Args:
        beta_values: List of beta values
        tau_values: List of tau values
        n_steps_values: List of n_steps values
        scenario_count: Number of scenarios to generate for each combination
        
    Returns:
        Dictionary mapping combination keys to lists of scenario IDs
    """
    scenarios = {}
    
    for beta in beta_values:
        for tau in tau_values:
            for n_steps in n_steps_values:
                # Create a key for this combination
                combo_key = f"beta{beta}_tau{tau}_steps{n_steps}"
                scenarios[combo_key] = []
                
                for i in range(scenario_count):
                    # Create a jammer scenario with this combination
                    jammer_id = create_jammer_scenario(
                        beta=beta,
                        tau=tau,
                        n_steps=n_steps,
                        scenario_id=f"jammer_{combo_key}_{i}"
                    )
                    if jammer_id:
                        scenarios[combo_key].append(jammer_id)
                    
                    # Only create normal scenario once per beta/n_steps combination
                    if i == 0:
                        normal_id = create_normal_scenario(
                            beta=beta,
                            n_steps=n_steps,
                            scenario_id=f"normal_beta{beta}_steps{n_steps}_{i}"
                        )
                        scenarios[combo_key].append(normal_id)
    
    return scenarios
