# utils/visualization.py
"""
Visualization tools for MIMO network simulation.
Provides functions for creating static and animated visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from typing import List, Optional, Tuple, Any
from IPython.display import HTML, display
import os

def visualize_network_snapshot(G: nx.Graph, jammer_position: Optional[np.ndarray] = None, 
                             jammer_active: bool = False, jammer_range: float = 0.35,
                             title: str = "", figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Create a static visualization of the network at a single time step.
    
    Args:
        G: NetworkX graph to visualize
        jammer_position: Position of jammer (if present)
        jammer_active: Whether jammer is active
        jammer_range: Range of jammer effect
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node positions and labels
    pos = nx.get_node_attributes(G, 'pos')
    labels = {node: node for node in G.nodes()}
    
    # Create edge labels with SINR and distance
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        # Using the attribute names from our code
        edge_labels[(u, v)] = f"SINR:{data['sinr']:.0f}dB\nDist:{data['distance']:.2f}km"
    
    # Draw nodes with different sizes and colors for AP vs Users
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=[150 if G.nodes[n]['type'] == 'AP' else 90 for n in G.nodes()],
        node_color=['red' if G.nodes[n]['type'] == 'AP' else 'green' for n in G.nodes()],
    )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=6
    )
    
    # Only draw jammer if provided
    if jammer_position is not None:
        jammer_color = 'purple' if jammer_active else 'gray'
        ax.scatter(jammer_position[0], jammer_position[1],
                  color=jammer_color, s=200, marker='*',
                  label='Jammer (Active)' if jammer_active else 'Jammer (Inactive)')
        
        # Draw jammer range
        circle = plt.Circle(jammer_position, jammer_range,
                          color=jammer_color, fill=False,
                          linestyle='--', alpha=0.3)
        ax.add_patch(circle)
    
    # Set plot properties
    ax.set_title(title)
    ax.set_xlim(0, 1)  # Assuming area_size=1.0
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    
    # Add appropriate legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=10, label='Access Points'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                  markersize=8, label='Mobile Users')
    ]
    if jammer_position is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='purple',
                      markersize=10, label='Jammer')
        )
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def create_network_animation(dynamic_graphs: List[nx.Graph], 
                           jammer_positions: Optional[List[np.ndarray]] = None,
                           jammer_states: Optional[List[bool]] = None,
                           jammer_range: float = 0.35,
                           title: str = "",
                           subtitle: str = "",
                           output_filename: Optional[str] = None,
                           figsize: Tuple[int, int] = (8, 8),
                           fps: int = 3) -> HTML:
    """
    Create an animation of the network over time.
    
    Args:
        dynamic_graphs: List of NetworkX graphs for each time step
        jammer_positions: List of jammer positions (or same position repeated)
        jammer_states: List of jammer activity states
        jammer_range: Range of jammer effect
        title: Main title for the animation
        subtitle: Subtitle with parameter information
        output_filename: Filename to save the animation
        figsize: Figure size
        fps: Frames per second for the animation
        
    Returns:
        IPython HTML object to display the animation
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set consistent jammer position/states if provided
    if jammer_positions is None and jammer_states is None:
        with_jammer = False
    else:
        with_jammer = True
        # If only one jammer position provided, repeat it for all frames
        if jammer_positions is not None and len(jammer_positions) == 1:
            jammer_position = jammer_positions[0]
            jammer_positions = [jammer_position] * len(dynamic_graphs)
    
    def update(frame):
        ax.clear()
        G = dynamic_graphs[frame]
        
        # Get node positions and labels
        pos = nx.get_node_attributes(G, 'pos')
        labels = {node: node for node in G.nodes()}
        
        # Create edge labels with SINR and distance
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = f"SINR:{data['sinr']:.0f}dB\nDist:{data['distance']:.2f}km"
        
        # Draw nodes with different sizes and colors for AP vs Users
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=[150 if G.nodes[n]['type'] == 'AP' else 90 for n in G.nodes()],
            node_color=['red' if G.nodes[n]['type'] == 'AP' else 'green' for n in G.nodes()],
        )
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=6
        )
        
        # Only draw jammer if in jammer mode
        if with_jammer:
            jammer_position = jammer_positions[frame] if jammer_positions else None
            jammer_active = jammer_states[frame] if jammer_states else False
            
            if jammer_position is not None:
                jammer_color = 'purple' if jammer_active else 'gray'
                ax.scatter(jammer_position[0], jammer_position[1],
                          color=jammer_color, s=200, marker='*',
                          label='Jammer (Active)' if jammer_active else 'Jammer (Inactive)')
                
                # Draw jammer range
                circle = plt.Circle(jammer_position, jammer_range,
                                  color=jammer_color, fill=False,
                                  linestyle='--', alpha=0.3)
                ax.add_patch(circle)
        
        # Set plot properties
        frame_title = title or ('Cell-Free MIMO Network with Jammer' if with_jammer else 'Normal Cell-Free MIMO Network')
        ax.set_title(f'{frame_title}\n{subtitle}\nStep {frame}')
        ax.set_xlim(0, 1)  # Assuming area_size=1.0
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
        
        # Add appropriate legend elements
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, label='Access Points'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                      markersize=8, label='Mobile Users')
        ]
        if with_jammer:
            legend_elements.append(
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='purple',
                          markersize=10, label='Jammer')
            )
        ax.legend(handles=legend_elements, loc='upper right')
        
        return ax.artists + ax.collections + ax.patches
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(dynamic_graphs))
    
    # Save as GIF if filename provided
    if output_filename:
        if not output_filename.endswith('.gif'):
            output_filename += '.gif'
        anim.save(output_filename, writer='pillow', fps=fps)
        plt.close()
        print(f"Animation saved as {output_filename}")
        return HTML(f'<img src="{output_filename}" width=800/>')
    
    return anim

def visualize_parameter_sweep(sweep_type: str = 'beta', n_steps: int = 40, 
                            save_dir: str = 'visualizations'):
    """
    Generate multiple visualizations with different parameter values.
    
    Args:
        sweep_type: 'beta' or 'tau'
        n_steps: Number of steps for each simulation
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if sweep_type == 'beta':
        # Sweep through different beta values (Rician parameter)
        beta_values = [0.0, 0.5, 1.0]  # Deterministic, Mixed, Fully Random
        for beta in beta_values:
            print(f"Generating visualization for beta={beta}...")
            
            # Import here to avoid circular imports
            from simulation.simulator import mimo_simulator
            
            # Run simulation with jammer
            graphs, success, jammer_states, jammer_position, _ = mimo_simulator.run_jammer_simulation(
                beta=beta,
                tau=5,  # Medium jammer activity
                n_steps=n_steps,
                n_frames=n_steps//10 if n_steps >= 10 else 1
            )
            
            if success:
                output_filename = os.path.join(save_dir, f"cf_mimo_jammer_beta{beta}_tau5.gif")
                subtitle = f"Beta={beta} (Rician Parameter), Tau=5 (Jammer Activity)"
                
                create_network_animation(
                    graphs,
                    jammer_positions=[jammer_position] * len(graphs),
                    jammer_states=jammer_states,
                    title="Cell-Free MIMO Network with Jammer",
                    subtitle=subtitle,
                    output_filename=output_filename
                )
    
    elif sweep_type == 'tau':
        # Sweep through different tau values (jammer activity)
        tau_values = [2, 5, 8]  # Low, Medium, High jammer activity
        beta = 0.5  # Fixed beta value
        
        for tau in tau_values:
            print(f"Generating visualization for tau={tau}...")
            
            # Import here to avoid circular imports
            from simulation.simulator import mimo_simulator
            
            # Run simulation
            graphs, success, jammer_states, jammer_position, _ = mimo_simulator.run_jammer_simulation(
                beta=beta,
                tau=tau,
                n_steps=n_steps,
                n_frames=n_steps//10 if n_steps >= 10 else 1
            )
            
            if success:
                output_filename = os.path.join(save_dir, f"cf_mimo_jammer_beta{beta}_tau{tau}.gif")
                subtitle = f"Beta={beta} (Rician Parameter), Tau={tau} (Jammer Activity)"
                
                create_network_animation(
                    graphs,
                    jammer_positions=[jammer_position] * len(graphs),
                    jammer_states=jammer_states,
                    title="Cell-Free MIMO Network with Jammer",
                    subtitle=subtitle,
                    output_filename=output_filename
                )
    
    else:
        print(f"Unknown sweep_type: {sweep_type}. Use 'beta' or 'tau'.")
