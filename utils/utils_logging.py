# utils/logging_utils.py
"""
Logging utilities for MIMO network simulation.
"""
import logging
import os
from typing import Optional

def setup_logger(log_file: str = "simulation.log", 
                log_level: int = logging.INFO,
                console_output: bool = True) -> logging.Logger:
    """
    Set up a logger for the simulation.
    
    Args:
        log_file: Path to the log file
        log_level: Logging level
        console_output: Whether to also log to console
        
    Returns:
        Configured logger
    """
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("mimo_simulation")
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def log_simulation_start(logger: logging.Logger, params: dict) -> None:
    """
    Log the start of a simulation with parameters.
    
    Args:
        logger: Logger to use
        params: Simulation parameters
    """
    logger.info("=" * 80)
    logger.info("Starting new simulation")
    logger.info("-" * 80)
    
    for key, value in params.items():
        logger.info(f"{key}: {value}")
    
    logger.info("-" * 80)

def log_simulation_progress(logger: logging.Logger, step: int, total_steps: int, 
                          metrics: Optional[dict] = None) -> None:
    """
    Log simulation progress.
    
    Args:
        logger: Logger to use
        step: Current step
        total_steps: Total number of steps
        metrics: Current metrics (optional)
    """
    progress_pct = (step / total_steps) * 100
    logger.info(f"Simulation progress: {step}/{total_steps} ({progress_pct:.1f}%)")
    
    if metrics:
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

def log_simulation_complete(logger: logging.Logger, simulation_id: str, 
                          runtime: float, success: bool = True) -> None:
    """
    Log the completion of a simulation.
    
    Args:
        logger: Logger to use
        simulation_id: ID of the simulation
        runtime: Simulation runtime in seconds
        success: Whether simulation completed successfully
    """
    status = "completed successfully" if success else "failed"
    logger.info("-" * 80)
    logger.info(f"Simulation {simulation_id} {status}")
    logger.info(f"Total runtime: {runtime:.2f} seconds")
    logger.info("=" * 80)

# Set up default logger
default_logger = setup_logger()
