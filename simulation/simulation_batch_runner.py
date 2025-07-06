# simulation/batch_runner.py
"""
Batch simulation runner for MIMO network simulation.
Handles running multiple simulations with different parameters.
"""
import os
import shutil
import logging
import time
import pickle
import tarfile
import zipfile
from threading import Lock
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures

import networkx as nx
from simulation.simulation_simulator import mimo_simulator
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lock for thread-safe operations
folder_lock = Lock()

def save_graph(graph: nx.Graph, file_path: str) -> None:
    """
    Save a NetworkX graph to disk using pickle.
    
    Args:
        graph: NetworkX graph to save
        file_path: Path to save the graph
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

def load_graph(file_path: str) -> nx.Graph:
    """
    Load a NetworkX graph from disk.
    
    Args:
        file_path: Path to the saved graph
        
    Returns:
        Loaded NetworkX graph
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compress_directory(directory_path: str, output_name: Optional[str] = None, 
                      method: str = 'zip') -> Optional[str]:
    """
    Compress a directory using Python's built-in libraries.
    
    Args:
        directory_path: Path to the directory to compress
        output_name: Name of the output archive (without extension)
        method: Compression method ('zip' or 'tar')
        
    Returns:
        Path to the compressed file or None if compression failed
    """
    if output_name is None:
        output_name = os.path.basename(directory_path)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine the base directory to ensure proper paths in archive
    base_dir = os.path.dirname(directory_path)
    dir_name = os.path.basename(directory_path)
    
    try:
        if method == 'zip':
            archive_path = f"{output_name}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, base_dir)
                        zipf.write(file_path, arcname)
        else:  # tar.gz
            archive_path = f"{output_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(directory_path, arcname=dir_name)
        
        return archive_path
    except Exception as e:
        logger.error(f"Error compressing directory {directory_path}: {str(e)}")
        return None

def process_single_simulation(params: Dict[str, Any]) -> Optional[str]:
    """
    Process a single simulation (normal or anomaly) based on parameters.
    This function is designed to be used with multiprocessing.
    
    Args:
        params: Dictionary of simulation parameters
        
    Returns:
        Folder path where results are saved, or None if simulation failed
    """
    simulation_type = params['simulation_type']
    beta_or_tau_value = params['beta_or_tau_value']
    param_type = params['param_type']
    sim_index = params['sim_index']
    n_steps_value = params['n_steps_value']
    base_folder = params['base_folder']
    
    # Create necessary directories
    folder_path = os.path.join(base_folder, f'{simulation_type}_graph_{sim_index}')
    temp_folder_path = os.path.join(config.RAMDISK_BASE, f'{simulation_type}_graph_{sim_index}')
    
    try:
        os.makedirs(temp_folder_path, exist_ok=True)
        
        if simulation_type == 'normal':
            # For normal simulations (without jammer)
            graphs, user_positions = mimo_simulator.run_normal_simulation(
                beta=beta_or_tau_value if param_type == 'beta' else 0,
                n_steps=n_steps_value
            )
        else:
            # For anomaly simulations (with jammer)
            n_frames = 8  # Default value
            graphs, success, jammer_states, jammer_position, user_positions = mimo_simulator.run_jammer_simulation(
                beta=beta_or_tau_value if param_type == 'beta' else 0,
                tau=beta_or_tau_value if param_type == 'tau' else 10,
                n_steps=n_steps_value,
                n_frames=n_frames
            )
            
            if not success:
                logger.warning(f"Failed to generate valid anomaly scenario for {simulation_type}_{sim_index}")
                shutil.rmtree(temp_folder_path, ignore_errors=True)
                return None
        
        # Save graphs temporarily in RAM
        for step, G in enumerate(graphs):
            save_graph(G, os.path.join(temp_folder_path, f"step_{step}.pkl"))
        
        # Create target directory
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        
        # Move from RAM to final folder
        with folder_lock:  # Use lock to avoid race conditions
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            shutil.move(temp_folder_path, folder_path)
        
        return folder_path
    
    except Exception as e:
        logger.error(f"Error in simulation {simulation_type}_{sim_index}: {str(e)}")
        # Cleanup in case of error
        if os.path.exists(temp_folder_path):
            shutil.rmtree(temp_folder_path, ignore_errors=True)
        return None

def run_simulations_for_parameter(param_value: float, param_type: str, 
                                normal_count: int, anomaly_count: int,
                                n_steps_value: Optional[int] = None, 
                                tau_value: int = 10, beta_value: float = 0,
                                is_combination: bool = False, 
                                combo_folder: Optional[str] = None,
                                use_parallel: bool = True) -> Tuple[str, str]:
    """
    Run simulations for a specific parameter value.
    
    Args:
        param_value: Value of the parameter (beta, tau, or n_steps)
        param_type: Type of parameter ('beta', 'tau', or 'n_steps')
        normal_count: Number of normal simulations to run
        anomaly_count: Number of anomaly simulations to run
        n_steps_value: Number of steps for simulation
        tau_value: Tau value for jammer activity
        beta_value: Beta value for simulations
        is_combination: Whether this is part of a parameter combination run
        combo_folder: Specific folder name for combination runs
        use_parallel: Whether to use parallel processing
        
    Returns:
        Tuple of (normal archive path, anomaly archive path)
    """
    # Set up paths based on parameter type and whether this is a combination run
    if is_combination and combo_folder:
        # Use the provided combination folder
        main_folder = "Combinations"
        param_folder = combo_folder
    else:
        # Use standard folder structure based on parameter type
        if param_type == 'beta':
            main_folder = 'Multiple_Beta_Values'
            param_folder = f'beta_{param_value:.2f}'
            steps = config.N_STEPS if n_steps_value is None else n_steps_value
        elif param_type == 'tau':
            main_folder = 'Multiple_Tau'
            param_folder = f'tau_{param_value:.2f}'
            steps = config.N_STEPS if n_steps_value is None else n_steps_value
            if beta_value == 0:
                main_folder = f'Multiple_Tau_With_Fading_Beta_{beta_value:.2f}'
            else:
                main_folder = f'Multiple_Tau_Without_Fading_Beta_{beta_value:.2f}'
        else:  # n_steps
            main_folder = 'Multiple_Observ_Windows'
            param_folder = f'N_steps_{param_value}'
            steps = param_value
            if beta_value > 0:
                main_folder = f'Multiple_Observ_Windows_Without_Fading_Beta_{beta_value:.2f}'
    
    base_folder_normal = os.path.join(main_folder, param_folder, 'normal_graphs')
    base_folder_anomaly = os.path.join(main_folder, param_folder, 'anomaly_graphs')
    
    # Ensure directories exist
    os.makedirs(base_folder_normal, exist_ok=True)
    os.makedirs(base_folder_anomaly, exist_ok=True)
    
    # Determine simulation steps
    if param_type == 'beta':
        steps = config.N_STEPS if n_steps_value is None else n_steps_value
    elif param_type == 'tau':
        steps = config.N_STEPS if n_steps_value is None else n_steps_value
    else:  # n_steps
        steps = param_value
    
    # Prepare parameters for processing
    normal_params = []
    anomaly_params = []
    
    for i in range(normal_count):
        if param_type == 'beta':
            normal_params.append({
                'simulation_type': 'normal',
                'beta_or_tau_value': param_value,
                'param_type': 'beta',
                'sim_index': i,
                'n_steps_value': steps,
                'base_folder': base_folder_normal
            })
        else:
            normal_params.append({
                'simulation_type': 'normal',
                'beta_or_tau_value': beta_value,
                'param_type': 'beta',
                'sim_index': i,
                'n_steps_value': steps,
                'base_folder': base_folder_normal
            })
    
    for i in range(anomaly_count):
        if param_type == 'beta':
            anomaly_params.append({
                'simulation_type': 'anomaly',
                'beta_or_tau_value': param_value,
                'param_type': 'beta',
                'sim_index': i,
                'n_steps_value': steps,
                'base_folder': base_folder_anomaly
            })
        elif param_type == 'tau':
            anomaly_params.append({
                'simulation_type': 'anomaly',
                'beta_or_tau_value': param_value,
                'param_type': 'tau',
                'sim_index': i,
                'n_steps_value': steps,
                'base_folder': base_folder_anomaly
            })
        else:  # n_steps
            anomaly_params.append({
                'simulation_type': 'anomaly',
                'beta_or_tau_value': tau_value,
                'param_type': 'tau',
                'sim_index': i,
                'n_steps_value': param_value,
                'base_folder': base_folder_anomaly
            })
    
    normal_completed = 0
    anomaly_completed = 0
    
    # Process simulations (either in parallel or sequentially)
    if use_parallel:
        # Determine the number of workers based on CPU cores
        max_workers = min(normal_count + anomaly_count, os.cpu_count() or 4)
        
        logger.info(f"Starting parallel processing with {max_workers} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit normal simulation jobs
            normal_futures = [executor.submit(process_single_simulation, params) 
                            for params in normal_params]
            
            # Wait for normal simulations to complete
            for future in concurrent.futures.as_completed(normal_futures):
                if future.result():
                    normal_completed += 1
                    if normal_completed % 5 == 0:
                        logger.info(f"Completed {normal_completed}/{normal_count} normal simulations")
            
            # Submit anomaly simulation jobs
            anomaly_futures = [executor.submit(process_single_simulation, params) 
                             for params in anomaly_params]
            
            # Wait for anomaly simulations to complete
            for future in concurrent.futures.as_completed(anomaly_futures):
                if future.result():
                    anomaly_completed += 1
                    if anomaly_completed % 5 == 0:
                        logger.info(f"Completed {anomaly_completed}/{anomaly_count} anomaly simulations")
    else:
        # Sequential processing
        logger.info(f"Starting {normal_count} normal simulations with {param_type}={param_value}")
        
        for params in normal_params:
            if process_single_simulation(params):
                normal_completed += 1
                if normal_completed % 5 == 0:
                    logger.info(f"Completed {normal_completed}/{normal_count} normal simulations")
        
        logger.info(f"Starting {anomaly_count} anomaly simulations with {param_type}={param_value}")
        
        for params in anomaly_params:
            if process_single_simulation(params):
                anomaly_completed += 1
                if anomaly_completed % 5 == 0:
                    logger.info(f"Completed {anomaly_completed}/{anomaly_count} anomaly simulations")
    
    logger.info(f"Completed {normal_completed}/{normal_count} normal simulations")
    logger.info(f"Completed {anomaly_completed}/{anomaly_count} anomaly simulations")
    
    # Create a directory for compressed output with same structure
    compressed_dir = os.path.join(main_folder, "compressed", param_folder)
    os.makedirs(compressed_dir, exist_ok=True)
    
    # Compress the directories with more specific names
    logger.info(f"Compressing normal graphs directory...")
    normal_archive = compress_directory(
        base_folder_normal,
        os.path.join(compressed_dir, f"{param_type}_{param_value}_normal")
    )
    
    logger.info(f"Compressing anomaly graphs directory...")
    anomaly_archive = compress_directory(
        base_folder_anomaly,
        os.path.join(compressed_dir, f"{param_type}_{param_value}_anomaly")
    )
    
    # Only remove the original directories if compression was successful
    if normal_archive:
        logger.info(f"Removing original normal graphs directory...")
        shutil.rmtree(base_folder_normal)
    else:
        logger.error(f"Failed to compress normal graphs directory. Original directory kept.")
    
    if anomaly_archive:
        logger.info(f"Removing original anomaly graphs directory...")
        shutil.rmtree(base_folder_anomaly)
    else:
        logger.error(f"Failed to compress anomaly graphs directory. Original directory kept.")
    
    logger.info(f"Completed processing for {param_type}={param_value}")
    logger.info(f"Compressed files saved to: {normal_archive or 'FAILED'} and {anomaly_archive or 'FAILED'}")
    
    return normal_archive or "", anomaly_archive or ""

def run_combinations(use_parallel: bool = True) -> List[str]:
    """
    Run various combinations of parameters:
    - Selected timestep values with selected tau values
    - All combinations with selected beta values (fading levels)
    
    Args:
        use_parallel: Whether to use parallel processing
        
    Returns:
        List of created combination folders
    """
    # Use subset of values for combinations to reduce total number
    selected_beta_values = [0, 1]  # Use only no-fading and full-fading
    selected_tau_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    selected_timestep_values = [80]
    
    # Track all created combination folders for logging
    created_combinations = []
    
    # For each beta value (with/without fading)
    for beta in selected_beta_values:
        beta_label = "With_Fading" if beta == 0 else "Without_Fading"
        logger.info(f"Starting combination simulations for {beta_label} (beta={beta})")
        
        # For each selected tau value, run simulations with different timesteps
        for tau in selected_tau_values:
            for n_steps in selected_timestep_values:
                # Create a unique folder name for this specific combination
                combo_folder = f"beta{beta:.1f}_tau{tau}_steps{n_steps}"
                logger.info(f"Processing combination: {combo_folder}")
                created_combinations.append(combo_folder)
                
                # Run the simulation with the specified combination of parameters
                run_simulations_for_parameter(
                    param_value=n_steps,
                    param_type='n_steps',
                    normal_count=config.DEFAULT_NORMAL_COUNT,
                    anomaly_count=config.DEFAULT_ANOMALY_COUNT,
                    tau_value=tau,
                    beta_value=beta,
                    is_combination=True,
                    combo_folder=combo_folder,
                    use_parallel=use_parallel
                )
    
    # Summary of all created combinations
    total_combinations = len(created_combinations)
    logger.info(f"Completed all {total_combinations} parameter combinations")
    logger.info(f"Combination folders created: {created_combinations}")
    
    return created_combinations

def run_all_simulations(use_parallel: bool = True) -> None:
    """
    Run all simulations with default values.
    
    Args:
        use_parallel: Whether to use parallel processing
    """
    start_time = time.time()
    
    # Run beta simulations
    logger.info(f"Running beta simulations for values: {config.DEFAULT_BETA_VALUES}")
    for beta in config.DEFAULT_BETA_VALUES:
        logger.info(f"Starting simulations for beta={beta}")
        run_simulations_for_parameter(
            beta, 'beta',
            normal_count=config.DEFAULT_NORMAL_COUNT,
            anomaly_count=config.DEFAULT_ANOMALY_COUNT,
            use_parallel=use_parallel
        )
    
    # Run tau simulations with both beta=0 and beta=1
    logger.info(f"Running tau simulations for values: {config.DEFAULT_TAU_VALUES}")
    for beta in [0, 1]:  # Run for both no fading and full fading
        beta_label = "with_fading" if beta == 0 else "without_fading"
        logger.info(f"Running tau simulations with beta={beta} ({beta_label})")
        for tau in config.DEFAULT_TAU_VALUES:
            logger.info(f"Starting simulations for tau={tau} with beta={beta}")
            run_simulations_for_parameter(
                tau, 'tau',
                normal_count=config.DEFAULT_NORMAL_COUNT,
                anomaly_count=config.DEFAULT_ANOMALY_COUNT,
                beta_value=beta,  # Explicitly set beta value
                use_parallel=use_parallel
            )
    
    # Run n_steps simulations with both beta=0 and beta=1
    logger.info(f"Running n_steps simulations for values: {config.DEFAULT_NSTEPS_VALUES}")
    for beta in [0, 1]:  # Run for both no fading and full fading
        beta_label = "with_fading" if beta == 0 else "without_fading"
        logger.info(f"Running n_steps simulations with beta={beta} ({beta_label})")
        for n_steps_val in config.DEFAULT_NSTEPS_VALUES:
            logger.info(f"Starting simulations for n_steps={n_steps_val} with beta={beta}")
            run_simulations_for_parameter(
                n_steps_val, 'n_steps',
                normal_count=config.DEFAULT_NORMAL_COUNT,
                anomaly_count=config.DEFAULT_ANOMALY_COUNT,
                tau_value=config.DEFAULT_TAU_VALUES[-1],  # Use the last tau value as default
                beta_value=beta,  # Explicitly set beta value
                use_parallel=use_parallel
            )
    
    # Run combinations
    logger.info("Running parameter combinations")
    run_combinations(use_parallel=use_parallel)
    
    total_time = time.time() - start_time
    logger.info(f"All simulations completed in {total_time:.2f} seconds")
