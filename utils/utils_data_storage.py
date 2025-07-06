# utils/data_storage.py
"""
Data storage utilities for MIMO network simulation.
Provides functions for saving and loading simulation data.
"""
import os
import pickle
import json
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super(NumpyEncoder, self).default(obj)

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

def save_simulation_results(
    simulation_id: str,
    graphs: List[nx.Graph],
    simulation_params: Dict[str, Any],
    user_positions: Optional[List[np.ndarray]] = None,
    jammer_states: Optional[List[bool]] = None,
    jammer_position: Optional[np.ndarray] = None,
    output_dir: str = 'simulation_results'
) -> str:
    """
    Save simulation results to disk.
    
    Args:
        simulation_id: Unique identifier for the simulation
        graphs: List of NetworkX graphs from simulation
        simulation_params: Dictionary of simulation parameters
        user_positions: List of user positions at each time step
        jammer_states: List of jammer activity states
        jammer_position: Position of the jammer
        output_dir: Directory to save results
        
    Returns:
        Path to the saved simulation directory
    """
    # Create simulation directory
    sim_dir = os.path.join(output_dir, simulation_id)
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(os.path.join(sim_dir, 'graphs'), exist_ok=True)
    
    # Save parameters
    params_file = os.path.join(sim_dir, 'params.json')
    with open(params_file, 'w') as f:
        # Include jammer position and states in parameters
        params_with_jammer = simulation_params.copy()
        if jammer_position is not None:
            params_with_jammer['jammer_position'] = jammer_position
        if jammer_states is not None:
            params_with_jammer['jammer_states'] = jammer_states
        
        json.dump(params_with_jammer, f, cls=NumpyEncoder, indent=2)
    
    # Save graphs
    for i, graph in enumerate(graphs):
        graph_file = os.path.join(sim_dir, 'graphs', f'step_{i}.pkl')
        save_graph(graph, graph_file)
    
    # Save user positions if provided
    if user_positions is not None:
        positions_file = os.path.join(sim_dir, 'user_positions.pkl')
        with open(positions_file, 'wb') as f:
            pickle.dump(user_positions, f, pickle.HIGHEST_PROTOCOL)
    
    return sim_dir

def load_simulation_results(simulation_id: str, 
                          input_dir: str = 'simulation_results'
                         ) -> Tuple[List[nx.Graph], Dict[str, Any], Optional[List[np.ndarray]]]:
    """
    Load simulation results from disk.
    
    Args:
        simulation_id: Unique identifier for the simulation
        input_dir: Directory containing simulation results
        
    Returns:
        Tuple of (list of graphs, simulation parameters, list of user positions)
    """
    sim_dir = os.path.join(input_dir, simulation_id)
    
    # Check if directory exists
    if not os.path.exists(sim_dir):
        raise FileNotFoundError(f"Simulation directory {sim_dir} not found")
    
    # Load parameters
    params_file = os.path.join(sim_dir, 'params.json')
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Process complex values in params
    for key, value in params.items():
        if isinstance(value, dict) and 'real' in value and 'imag' in value:
            params[key] = complex(value['real'], value['imag'])
    
    # Convert jammer position back to numpy array if present
    if 'jammer_position' in params:
        params['jammer_position'] = np.array(params['jammer_position'])
    
    # Load graphs
    graphs_dir = os.path.join(sim_dir, 'graphs')
    graph_files = sorted(
        [f for f in os.listdir(graphs_dir) if f.startswith('step_')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    
    graphs = []
    for graph_file in graph_files:
        graph_path = os.path.join(graphs_dir, graph_file)
        graphs.append(load_graph(graph_path))
    
    # Load user positions if available
    user_positions = None
    positions_file = os.path.join(sim_dir, 'user_positions.pkl')
    if os.path.exists(positions_file):
        with open(positions_file, 'rb') as f:
            user_positions = pickle.load(f)
    
    return graphs, params, user_positions

def extract_simulation_metrics(graphs: List[nx.Graph]) -> Dict[str, List[float]]:
    """
    Extract key metrics from simulation graphs.
    
    Args:
        graphs: List of NetworkX graphs from simulation
        
    Returns:
        Dictionary containing time series of various metrics
    """
    # Initialize metrics dictionary
    metrics = {
        'time_steps': list(range(len(graphs))),
        'connection_count': [],
        'avg_sinr': [],
        'min_sinr': [],
        'max_sinr': [],
        'avg_distance': [],
        'avg_interference': [],
        'avg_jammer_interference': []
    }
    
    # Extract metrics for each time step
    for graph in graphs:
        # Count connections
        edge_count = graph.number_of_edges()
        metrics['connection_count'].append(edge_count)
        
        # SINR statistics
        sinr_values = [data.get('sinr', 0) for _, _, data in graph.edges(data=True)]
        if sinr_values:
            metrics['avg_sinr'].append(sum(sinr_values) / len(sinr_values))
            metrics['min_sinr'].append(min(sinr_values))
            metrics['max_sinr'].append(max(sinr_values))
        else:
            metrics['avg_sinr'].append(0)
            metrics['min_sinr'].append(0)
            metrics['max_sinr'].append(0)
        
        # Distance statistics
        distance_values = [data.get('distance', 0) for _, _, data in graph.edges(data=True)]
        metrics['avg_distance'].append(
            sum(distance_values) / len(distance_values) if distance_values else 0
        )
        
        # Interference statistics
        interference_values = [data.get('interference', 0) for _, _, data in graph.edges(data=True)]
        metrics['avg_interference'].append(
            sum(interference_values) / len(interference_values) if interference_values else 0
        )
        
        # Jammer interference statistics
        jammer_interference_values = [
            data.get('jammer_interference', 0) for _, _, data in graph.edges(data=True)
        ]
        metrics['avg_jammer_interference'].append(
            sum(jammer_interference_values) / len(jammer_interference_values) 
            if jammer_interference_values else 0
        )
    
    return metrics

def generate_simulation_summary(
    simulation_id: str, 
    metrics: Dict[str, List[float]],
    params: Dict[str, Any],
    output_dir: str = 'simulation_results'
) -> str:
    """
    Generate and save a summary of simulation results.
    
    Args:
        simulation_id: Unique identifier for the simulation
        metrics: Dictionary of simulation metrics
        params: Simulation parameters
        output_dir: Directory to save summary
        
    Returns:
        Path to the saved summary file
    """
    sim_dir = os.path.join(output_dir, simulation_id)
    summary_file = os.path.join(sim_dir, 'summary.json')
    
    # Create summary dictionary
    summary = {
        'simulation_id': simulation_id,
        'parameters': params,
        'metrics_summary': {
            'avg_connection_count': sum(metrics['connection_count']) / len(metrics['connection_count']),
            'avg_sinr_overall': sum(metrics['avg_sinr']) / len(metrics['avg_sinr']),
            'min_sinr_overall': min(metrics['min_sinr']),
            'max_sinr_overall': max(metrics['max_sinr']),
            'avg_distance_overall': sum(metrics['avg_distance']) / len(metrics['avg_distance']),
            'avg_interference_overall': sum(metrics['avg_interference']) / len(metrics['avg_interference']),
            'avg_jammer_interference_overall': (
                sum(metrics['avg_jammer_interference']) / len(metrics['avg_jammer_interference'])
            )
        }
    }
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, cls=NumpyEncoder, indent=2)
    
    return summary_file
