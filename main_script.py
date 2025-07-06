# main.py
"""
Main entry point for MIMO network simulation.
"""
import argparse
import time
import logging
import os
import sys
from typing import Optional

# ...existing code...
from models.models_channel import channel_model
from models.models_network import network_model
from models.models_mobility import mobility_model
from models.models_jammer import jammer_model
from simulation.simulation_simulator import mimo_simulator
from simulation.simulation_batch_runner import run_simulations_for_parameter, run_combinations, run_all_simulations
from simulation.simulation_scenarios import (
    create_normal_scenario, create_jammer_scenario, 
    create_parameter_sweep_scenarios, create_combination_scenarios
)
from utils.utils_visualization import *




from utils.utils_visualization import (
    visualize_network_snapshot, create_network_animation, visualize_parameter_sweep
)
from utils.utils_logging import setup_logger
import config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MIMO Network Simulation")
    
    # Main simulation mode
    parser.add_argument(
        "--mode", 
        choices=["single", "batch", "sweep", "combinations", "visualize"],
        default="single",
        help="Simulation mode"
    )
    
    # Single simulation parameters
    parser.add_argument("--beta", type=float, default=0.0, help="Rician fading parameter")
    parser.add_argument("--tau", type=int, default=5, help="Jammer active duration")
    parser.add_argument("--steps", type=int, default=80, help="Number of simulation steps")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--with-jammer", action="store_true", help="Include jammer in simulation")
    
    # Batch simulation parameters
    parser.add_argument("--normal-count", type=int, default=2, help="Number of normal simulations")
    parser.add_argument("--anomaly-count", type=int, default=2, help="Number of anomaly simulations")
    parser.add_argument("--use-parallel", action="store_true", help="Use parallel processing")
    
    # Parameter sweep parameters
    parser.add_argument(
        "--sweep-type", 
        choices=["beta", "tau", "n_steps"],
        default="beta",
        help="Parameter to sweep"
    )
    parser.add_argument(
        "--sweep-values", 
        type=float, 
        nargs="+", 
        help="Values to use for parameter sweep"
    )
    
    # Visualization parameters
    parser.add_argument(
        "--vis-type",
        choices=["snapshot", "animation", "sweep"],
        default="animation",
        help="Type of visualization to create"
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--scenario-id", type=str, help="Scenario ID to visualize")
    
    # Logging parameters
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--log-file", type=str, default="simulation.log", help="Log file path")
    
    return parser.parse_args()

def run_single_simulation(args):
    """Run a single simulation based on arguments."""
    logger.info("Running single simulation")
    
    if args.with_jammer:
        logger.info(f"Creating jammer scenario with beta={args.beta}, tau={args.tau}, steps={args.steps}")
        scenario_id = create_jammer_scenario(
            beta=args.beta,
            tau=args.tau,
            n_steps=args.steps,
            n_frames=args.frames
        )
        if scenario_id:
            logger.info(f"Jammer scenario created successfully: {scenario_id}")
            return scenario_id
        else:
            logger.error("Failed to create jammer scenario")
            return None
    else:
        logger.info(f"Creating normal scenario with beta={args.beta}, steps={args.steps}")
        scenario_id = create_normal_scenario(
            beta=args.beta,
            n_steps=args.steps
        )
        logger.info(f"Normal scenario created successfully: {scenario_id}")
        return scenario_id

def run_parameter_sweep(args):
    """Run a parameter sweep based on arguments."""
    logger.info(f"Running parameter sweep for {args.sweep_type}")
    
    # Use provided values or defaults
    values = args.sweep_values
    
    scenarios = create_parameter_sweep_scenarios(
        sweep_type=args.sweep_type,
        values=values,
        n_steps=args.steps,
        scenario_count=1
    )
    
    logger.info(f"Parameter sweep completed. Created scenarios: {scenarios}")
    return scenarios

def run_batch_mode(args):
    """Run batch simulations based on arguments."""
    logger.info("Running batch simulations")
    
    if args.sweep_type == "beta":
        param_value = args.beta
        param_type = "beta"
    elif args.sweep_type == "tau":
        param_value = args.tau
        param_type = "tau"
    elif args.sweep_type == "n_steps":
        param_value = args.steps
        param_type = "n_steps"
    
    normal_archive, anomaly_archive = run_simulations_for_parameter(
        param_value=param_value,
        param_type=param_type,
        normal_count=args.normal_count,
        anomaly_count=args.anomaly_count,
        n_steps_value=args.steps,
        tau_value=args.tau,
        beta_value=args.beta,
        use_parallel=args.use_parallel
    )
    
    logger.info("Batch simulations completed")
    logger.info(f"Normal archive: {normal_archive}")
    logger.info(f"Anomaly archive: {anomaly_archive}")

def run_combinations_mode(args):
    """Run combinations of parameters."""
    logger.info("Running parameter combinations")
    
    # For smaller test runs, use just a few values
    beta_values = [0.0, 1.0]
    tau_values = [2, 5, 8]
    n_steps_values = [80]
    
    scenarios = create_combination_scenarios(
        beta_values=beta_values,
        tau_values=tau_values,
        n_steps_values=n_steps_values,
        scenario_count=1
    )
    
    logger.info(f"Combinations completed. Created scenarios: {scenarios}")
    return scenarios

def run_visualization(args):
    """Run visualization based on arguments."""
    logger.info(f"Running visualization of type {args.vis_type}")
    
    if args.vis_type == "sweep":
        visualize_parameter_sweep(
            sweep_type=args.sweep_type,
            n_steps=args.steps,
            save_dir=args.output_dir
        )
        logger.info(f"Parameter sweep visualization completed. Results saved to {args.output_dir}")
    
    elif args.vis_type == "animation" and args.scenario_id:
        # First would need to load the scenario from disk
        # This would require implementing a scenario loading function
        logger.warning("Animation of saved scenarios not implemented yet")
    
    elif args.vis_type == "snapshot" and args.scenario_id:
        # Similar to animation, would need to load scenario first
        logger.warning("Snapshot of saved scenarios not implemented yet")
    
    else:
        # Run a new simulation and visualize it
        scenario_id = run_single_simulation(args)
        if scenario_id:
            logger.info(f"Visualization for scenario {scenario_id} not implemented yet")
        else:
            logger.error("Failed to create scenario for visualization")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    global logger
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(args.log_file, log_level)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run appropriate mode
    start_time = time.time()
    
    try:
        if args.mode == "single":
            scenario_id = run_single_simulation(args)
            if scenario_id:
                logger.info(f"Single simulation completed successfully: {scenario_id}")
            else:
                logger.error("Single simulation failed")
                return 1
        
        elif args.mode == "batch":
            run_batch_mode(args)
        
        elif args.mode == "sweep":
            scenarios = run_parameter_sweep(args)
            if scenarios:
                logger.info(f"Parameter sweep completed successfully")
            else:
                logger.error("Parameter sweep failed")
                return 1
        
        elif args.mode == "combinations":
            scenarios = run_combinations_mode(args)
            if scenarios:
                logger.info(f"Combinations mode completed successfully")
            else:
                logger.error("Combinations mode failed")
                return 1
        
        elif args.mode == "visualize":
            run_visualization(args)
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
    
    except Exception as e:
        logger.exception(f"Error in {args.mode} mode: {str(e)}")
        return 1
    
    runtime = time.time() - start_time
    logger.info(f"Total runtime: {runtime:.2f} seconds")
    return 0

if __name__ == "__main__":
    sys.exit(main())
