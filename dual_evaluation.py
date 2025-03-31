"""
GPU-Enhanced Visualization Module

This module provides functions for creating visualizations of agent performance
with a focus on GPU resource utilization. It can be used independently to visualize
results from previously run experiments.

Key visualizations include:
- GPU vs CPU memory usage comparison
- Win rate vs GPU resource trade-offs
- Resource distribution radar charts
- Performance vs resource Pareto analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import datetime


def load_evaluation_data(eval_dir):
    """
    Load evaluation data from summary files
    
    Args:
        eval_dir: Directory containing evaluation summaries
        
    Returns:
        tuple: (fixed_episodes_data, fixed_resource_data)
    """
    fixed_episodes_data = {}
    fixed_resource_data = {}
    
    # Look for evaluation summary files
    episode_files = glob.glob(os.path.join(eval_dir, "fixed_episodes_*/evaluation_summary.json"))
    resource_files = glob.glob(os.path.join(eval_dir, "fixed_resource_*/evaluation_summary.json"))
    
    # Load fixed episodes data
    if episode_files:
        latest_episode_file = max(episode_files, key=os.path.getctime)
        print(f"Loading fixed episodes data from: {latest_episode_file}")
        
        try:
            with open(latest_episode_file, 'r') as f:
                summary = json.load(f)
                fixed_episodes_data = summary.get("results", {})
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading fixed episodes data: {e}")
    
    # Load fixed resource data
    if resource_files:
        latest_resource_file = max(resource_files, key=os.path.getctime)
        print(f"Loading fixed resource data from: {latest_resource_file}")
        
        try:
            with open(latest_resource_file, 'r') as f:
                summary = json.load(f)
                fixed_resource_data = summary.get("results", {})
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading fixed resource data: {e}")
    
    return fixed_episodes_data, fixed_resource_data


def load_cnp_metrics(eval_dir):
    """
    Load CNP metrics from metrics files
    
    Args:
        eval_dir: Directory containing CNP metrics
        
    Returns:
        dict: CNP metrics data
    """
    cnp_metrics = {}
    
    # Look for CNP metrics files
    metrics_files = glob.glob(os.path.join(eval_dir, "cnp_metrics_*/cnp_metrics.json"))
    
    if metrics_files:
        latest_metrics_file = max(metrics_files, key=os.path.getctime)
        print(f"Loading CNP metrics from: {latest_metrics_file}")
        
        try:
            with open(latest_metrics_file, 'r') as f:
                cnp_metrics = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading CNP metrics: {e}")
    
    return cnp_metrics


def create_agent_dataframe(fixed_episodes_data, fixed_resource_data, cnp_metrics=None):
    """
    Create a consolidated DataFrame for visualization
    
    Args:
        fixed_episodes_data: Results from fixed episodes evaluation
        fixed_resource_data: Results from fixed resource evaluation
        cnp_metrics: Optional CNP metrics data
        
    Returns:
        DataFrame: Consolidated data for visualization
    """
    agents = []
    
    # Process fixed episodes data
    for agent_name, metrics in fixed_episodes_data.items():
        win_rate = metrics.get("win_rate", 0) * 100
        avg_time_ms = metrics.get("agent_performance", {}).get(agent_name, {}).get("avg_time_ms", 0)
        memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_memory_mb", 0)
        
        # Get GPU metrics if available
        gpu_memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_gpu_memory_mb", 0)
        gpu_percent = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_gpu_percent", 0)
        
        # Get CNP metrics if available
        cnp_value = None
        cpu_only_cnp = None
        gpu_focused_cnp = None
        resource_distribution = None
        
        if cnp_metrics and agent_name in cnp_metrics:
            cnp_value = cnp_metrics[agent_name].get("fixed_episodes_cnp", 0)
            cpu_only_cnp = cnp_metrics[agent_name].get("fixed_episodes_cpu_only_cnp", 0)
            gpu_focused_cnp = cnp_metrics[agent_name].get("fixed_episodes_gpu_focused_cnp", 0)
            resource_distribution = cnp_metrics[agent_name].get("fixed_episodes_resource_distribution", {})
        
        agent_data = {
            "Agent": agent_name,
            "Evaluation": "Fixed Episodes",
            "Win Rate (%)": win_rate,
            "Avg Time (ms)": avg_time_ms,
            "Memory (MB)": memory_mb,
            "GPU Memory (MB)": gpu_memory_mb,
            "GPU Utilization (%)": gpu_percent,
            "Standard CNP": cnp_value,
            "CPU-only CNP": cpu_only_cnp,
            "GPU-focused CNP": gpu_focused_cnp
        }
        
        # Add resource distribution if available
        if resource_distribution:
            agent_data["CPU Resource %"] = resource_distribution.get("cpu_percent", 50)
            agent_data["GPU Resource %"] = resource_distribution.get("gpu_percent", 50)
        
        agents.append(agent_data)
    
    # Process fixed resource data
    for agent_name, metrics in fixed_resource_data.items():
        win_rate = metrics.get("win_rate", 0) * 100
        avg_time_ms = metrics.get("agent_performance", {}).get(agent_name, {}).get("avg_time_ms", 0)
        memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_memory_mb", 0)
        episodes = metrics.get("episodes_completed", 0)
        
        # Get GPU metrics if available
        gpu_memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_gpu_memory_mb", 0)
        gpu_percent = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_gpu_percent", 0)
        
        # Get CNP metrics if available
        cnp_value = None
        cpu_only_cnp = None
        gpu_focused_cnp = None
        resource_distribution = None
        
        if cnp_metrics and agent_name in cnp_metrics:
            cnp_value = cnp_metrics[agent_name].get("fixed_resource_cnp", 0)
            cpu_only_cnp = cnp_metrics[agent_name].get("fixed_resource_cpu_only_cnp", 0)
            gpu_focused_cnp = cnp_metrics[agent_name].get("fixed_resource_gpu_focused_cnp", 0)
            resource_distribution = cnp_metrics[agent_name].get("fixed_resource_resource_distribution", {})
        
        agent_data = {
            "Agent": agent_name,
            "Evaluation": "Fixed Resource",
            "Win Rate (%)": win_rate,
            "Avg Time (ms)": avg_time_ms,
            "Memory (MB)": memory_mb,
            "GPU Memory (MB)": gpu_memory_mb,
            "GPU Utilization (%)": gpu_percent,
            "Episodes": episodes,
            "Standard CNP": cnp_value,
            "CPU-only CNP": cpu_only_cnp,
            "GPU-focused CNP": gpu_focused_cnp
        }
        
        # Add resource distribution if available
        if resource_distribution:
            agent_data["CPU Resource %"] = resource_distribution.get("cpu_percent", 50)
            agent_data["GPU Resource %"] = resource_distribution.get("gpu_percent", 50)
        
        agents.append(agent_data)
    
    return pd.DataFrame(agents)


def create_gpu_pareto_chart(df, output_dir):
    """
    Create Pareto charts with GPU metrics
    
    Args:
        df: DataFrame containing agent performance data
        output_dir: Directory to save output visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. GPU Memory vs Win Rate
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=df,
        x="GPU Memory (MB)",
        y="Win Rate (%)",
        hue="Agent",
        style="Evaluation" if "Evaluation" in df.columns else None,
        size="Avg Time (ms)",
        sizes=(100, 500),
        alpha=0.7
    )
    
    plt.title("GPU Memory vs Win Rate Trade-off", fontsize=16)
    plt.xlabel("GPU Memory Consumption (MB)", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for each point
    for i, row in df.iterrows():
        plt.annotate(
            f"{row['Agent']} ({'FE' if row.get('Evaluation') == 'Fixed Episodes' else 'FR'})" if 'Evaluation' in row.columns else row['Agent'],
            (row['GPU Memory (MB)'], row['Win Rate (%)']),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )
    
    plt.savefig(os.path.join(output_dir, "gpu_memory_vs_winrate.png"), dpi=300)
    plt.close()
    
    # 2. GPU Utilization vs Win Rate
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=df,
        x="GPU Utilization (%)",
        y="Win Rate (%)",
        hue="Agent",
        style="Evaluation" if "Evaluation" in df.columns else None,
        size="Avg Time (ms)",
        sizes=(100, 500),
        alpha=0.7
    )
    
    plt.title("GPU Utilization vs Win Rate Trade-off", fontsize=16)
    plt.xlabel("GPU Utilization (%)", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for each point
    for i, row in df.iterrows():
        plt.annotate(
            f"{row['Agent']} ({'FE' if row.get('Evaluation') == 'Fixed Episodes' else 'FR'})" if 'Evaluation' in row.columns else row['Agent'],
            (row['GPU Utilization (%)'], row['Win Rate (%)']),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )
    
    plt.savefig(os.path.join(output_dir, "gpu_utilization_vs_winrate.png"), dpi=300)
    plt.close()
    
    # 3. 3D visualization with CPU Memory, GPU Memory, and Win Rate
    try:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            df['Memory (MB)'],
            df['GPU Memory (MB)'],
            df['Win Rate (%)'],
            c=df['GPU Utilization (%)'],
            cmap='viridis',
            s=df['Avg Time (ms)'] / 2,
            alpha=0.7
        )
        
        ax.set_xlabel('CPU Memory (MB)', fontsize=12)
        ax.set_ylabel('GPU Memory (MB)', fontsize=12)
        ax.set_zlabel('Win Rate (%)', fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('GPU Utilization (%)', fontsize=12)
        
        # Add annotations
        for i, row in df.iterrows():
            agent_label = row['Agent']
            if 'Evaluation' in df.columns:
                agent_label += f" ({'FE' if row['Evaluation'] == 'Fixed Episodes' else 'FR'})"
                
            ax.text(
                row['Memory (MB)'],
                row['GPU Memory (MB)'],
                row['Win Rate (%)'],
                agent_label,
                fontsize=9
            )
        
        plt.title("3D Resource Usage vs Performance", fontsize=16)
        plt.savefig(os.path.join(output_dir, "3d_gpu_cpu_memory_winrate.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating 3D plot: {e}")


def create_gpu_enhanced_radar_chart(df, output_dir):
    """
    Create radar charts including GPU metrics
    
    Args:
        df: DataFrame containing agent performance data
        output_dir: Directory to save output visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If we have multiple evaluation types, create a grouped DataFrame
    if 'Evaluation' in df.columns:
        # Group by Agent and Evaluation
        grouped_df = df.groupby(['Agent', 'Evaluation']).mean().reset_index()
    else:
        # Group by Agent only
        grouped_df = df.groupby('Agent').mean().reset_index()
    
    # Normalize metrics for radar chart
    df_norm = grouped_df.copy()
    
    # Metrics to normalize
    metrics = [
        'Win Rate (%)', 
        'Avg Time (ms)', 
        'Memory (MB)', 
        'GPU Memory (MB)', 
        'GPU Utilization (%)'
    ]
    
    # Determine which metrics should be maximized vs minimized
    maximize_metrics = ['Win Rate (%)']
    minimize_metrics = ['Avg Time (ms)', 'Memory (MB)', 'GPU Memory (MB)', 'GPU Utilization (%)']
    
    # Normalize each metric to [0, 1] range
    for col in metrics:
        max_val = grouped_df[col].max()
        min_val = grouped_df[col].min()
        range_val = max_val - min_val
        
        if range_val > 0:
            if col in maximize_metrics:
                # For metrics we want to maximize, higher values should be closer to 1
                df_norm[col] = (grouped_df[col] - min_val) / range_val
            else:
                # For metrics we want to minimize, lower values should be closer to 1
                df_norm[col] = 1 - ((grouped_df[col] - min_val) / range_val)
        else:
            df_norm[col] = 0.5  # Default if no variation
    
    # Create radar chart
    plt.figure(figsize=(12, 10))
    
    # Categories for radar chart
    categories = [
        'Win Rate', 
        'Time Efficiency', 
        'Memory Efficiency', 
        'GPU Memory Efficiency', 
        'GPU Utilization Efficiency'
    ]
    
    # Number of categories
    N = len(categories)
    
    # Angles for each category (in radians)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Draw one line per agent/evaluation combination with different colors
    if 'Evaluation' in df_norm.columns:
        # Create a unique entry for each Agent/Evaluation pair
        for i, row in df_norm.iterrows():
            agent = row['Agent']
            eval_type = row['Evaluation']
            label = f"{agent} ({eval_type})"
            
            # Values for each category
            values = [
                row['Win Rate (%)'],
                row['Avg Time (ms)'],
                row['Memory (MB)'],
                row['GPU Memory (MB)'],
                row['GPU Utilization (%)']
            ]
            
            # Close the loop
            values += values[:1]
            
            # Plot the agent's data
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=label)
            ax.fill(angles, values, alpha=0.1)
    else:
        # Draw one line per agent
        for agent in df_norm['Agent'].unique():
            agent_data = df_norm[df_norm['Agent'] == agent]
            
            # Values for each category
            values = [
                agent_data['Win Rate (%)'].values[0],
                agent_data['Avg Time (ms)'].values[0],
                agent_data['Memory (MB)'].values[0],
                agent_data['GPU Memory (MB)'].values[0],
                agent_data['GPU Utilization (%)'].values[0]
            ]
            
            # Close the loop
            values += values[:1]
            
            # Plot the agent's data
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=agent)
            ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Add axis lines
    ax.set_rlabel_position(0)
    
    # Set y-ticks
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # Add a title
    plt.title('GPU-Enhanced Agent Performance Comparison', size=16)
    
    # Add explanatory annotations
    plt.annotate('Higher is better for all metrics', xy=(0, 0), xytext=(-0.2, -0.15), 
                textcoords='axes fraction', ha='center', fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_enhanced_radar_chart.png"), dpi=300)
    plt.close()


def create_cnp_comparison_chart(df, output_dir):
    """
    Create bar charts comparing different CNP calculation methods
    
    Args:
        df: DataFrame containing agent performance and CNP data
        output_dir: Directory to save output visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for rows with CNP data
    cnp_df = df[df['Standard CNP'].notna()].copy()
    
    if cnp_df.empty:
        print("No CNP data available for visualization")
        return
    
    # If we have evaluation types, create separate charts for each
    if 'Evaluation' in cnp_df.columns:
        for eval_type in cnp_df['Evaluation'].unique():
            eval_df = cnp_df[cnp_df['Evaluation'] == eval_type]
            
            # Create grouped bar chart for CNP comparison
            plt.figure(figsize=(14, 8))
            
            # Set width of bars
            bar_width = 0.25
            index = np.arange(len(eval_df['Agent']))
            
            # Create bars
            plt.bar(index, eval_df['Standard CNP'], bar_width, label='Standard CNP', color='skyblue')
            plt.bar(index + bar_width, eval_df['CPU-only CNP'], bar_width, label='CPU-only CNP', color='lightgreen')
            plt.bar(index + 2 * bar_width, eval_df['GPU-focused CNP'], bar_width, label='GPU-focused CNP', color='salmon')
            
            # Add labels and title
            plt.xlabel('Agent', fontsize=14)
            plt.ylabel('CNP Value', fontsize=14)
            plt.title(f'Comparison of CNP Calculation Methods - {eval_type}', fontsize=16)
            plt.xticks(index + bar_width, eval_df['Agent'])
            plt.legend()
            
            # Add value labels on top of bars
            for i, value in enumerate(eval_df['Standard CNP']):
                plt.text(i, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
            
            for i, value in enumerate(eval_df['CPU-only CNP']):
                plt.text(i + bar_width, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
            
            for i, value in enumerate(eval_df['GPU-focused CNP']):
                plt.text(i + 2 * bar_width, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"cnp_comparison_{eval_type}.png"), dpi=300)
            plt.close()
    else:
        # Create grouped bar chart for CNP comparison
        plt.figure(figsize=(14, 8))
        
        # Set width of bars
        bar_width = 0.25
        index = np.arange(len(cnp_df['Agent']))
        
        # Create bars
        plt.bar(index, cnp_df['Standard CNP'], bar_width, label='Standard CNP', color='skyblue')
        plt.bar(index + bar_width, cnp_df['CPU-only CNP'], bar_width, label='CPU-only CNP', color='lightgreen')
        plt.bar(index + 2 * bar_width, cnp_df['GPU-focused CNP'], bar_width, label='GPU-focused CNP', color='salmon')
        
        # Add labels and title
        plt.xlabel('Agent', fontsize=14)
        plt.ylabel('CNP Value', fontsize=14)
        plt.title('Comparison of CNP Calculation Methods', fontsize=16)
        plt.xticks(index + bar_width, cnp_df['Agent'])
        plt.legend()
        
        # Add value labels on top of bars
        for i, value in enumerate(cnp_df['Standard CNP']):
            plt.text(i, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
        
        for i, value in enumerate(cnp_df['CPU-only CNP']):
            plt.text(i + bar_width, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
        
        for i, value in enumerate(cnp_df['GPU-focused CNP']):
            plt.text(i + 2 * bar_width, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cnp_comparison.png"), dpi=300)
        plt.close()
    
    # Create pie charts showing resource distribution for each agent
    for i, row in cnp_df.iterrows():
        if "CPU Resource %" not in row or "GPU Resource %" not in row:
            continue
            
        agent = row['Agent']
        eval_label = f"_{row['Evaluation']}" if 'Evaluation' in cnp_df.columns else ""
        
        plt.figure(figsize=(8, 8))
        
        # Data for pie chart
        resources = ['CPU Resources', 'GPU Resources']
        sizes = [row['CPU Resource %'], row['GPU Resource %']]
        colors = ['skyblue', 'lightcoral']
        explode = (0, 0.1)  # Explode the GPU slice for emphasis
        
        # Create pie chart
        plt.pie(sizes, explode=explode, labels=resources, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        
        # Add a title
        plt.title(f'Resource Distribution for {agent}{eval_label}', size=16)
        
        # Save the figure
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.savefig(os.path.join(output_dir, f"{agent}{eval_label}_resource_distribution.png"), dpi=300)
        plt.close()


def create_gpu_impact_visualization(df, output_dir):
    """
    Create visualizations showing the impact of GPU usage on performance
    
    Args:
        df: DataFrame with performance and resource data
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter plot of GPU utilization vs performance
    plt.figure(figsize=(12, 8))
    
    # Create a color gradient based on GPU memory
    scatter = plt.scatter(
        df['GPU Utilization (%)'], 
        df['Win Rate (%)'],
        c=df['GPU Memory (MB)'],
        s=df['Avg Time (ms)'] / 2,
        cmap='viridis',
        alpha=0.8
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('GPU Memory (MB)', fontsize=12)
    
    # Add labels and title
    plt.xlabel('GPU Utilization (%)', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.title('Impact of GPU Usage on Performance', fontsize=16)
    
    # Add annotations for each point
    for i, row in df.iterrows():
        label = row['Agent']
        if 'Evaluation' in df.columns:
            label += f" ({'FE' if row['Evaluation'] == 'Fixed Episodes' else 'FR'})"
            
        plt.annotate(
            label, 
            (row['GPU Utilization (%)'], row['Win Rate (%)']),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11
        )
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_impact_performance.png"), dpi=300)
    plt.close()
    
    # 2. Bar chart of GPU Memory vs Win Rate with CPU Memory stacked
    plt.figure(figsize=(14, 8))
    
    # Create a grouped DataFrame
    if 'Evaluation' in df.columns:
        grouped_df = df.groupby(['Agent', 'Evaluation']).mean().reset_index()
        
        # Create separate charts for each evaluation type
        for eval_type in grouped_df['Evaluation'].unique():
            eval_df = grouped_df[grouped_df['Evaluation'] == eval_type]
            
            # Sort by win rate
            eval_df = eval_df.sort_values('Win Rate (%)', ascending=False)
            
            # Create positions for bars
            x = np.arange(len(eval_df))
            
            # Create stacked bars for CPU and GPU memory
            plt.figure(figsize=(14, 8))
            plt.bar(x, eval_df['Memory (MB)'], label='CPU Memory (MB)', color='skyblue')
            plt.bar(x, eval_df['GPU Memory (MB)'], bottom=eval_df['Memory (MB)'], 
                    label='GPU Memory (MB)', color='orange')
            
            # Plot win rate line on secondary y-axis
            ax2 = plt.twinx()
            ax2.plot(x, eval_df['Win Rate (%)'], 'ro-', linewidth=2, label='Win Rate (%)')
            
            # Set labels and title
            plt.xlabel('Agent', fontsize=14)
            plt.ylabel('Memory Usage (MB)', fontsize=14)
            ax2.set_ylabel('Win Rate (%)', fontsize=14, color='r')
            plt.title(f'Memory Usage Distribution vs. Performance - {eval_type}', fontsize=16)
            
            # Set x-tick labels to agent names
            plt.xticks(x, eval_df['Agent'])
            
            # Add legends
            plt.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"memory_distribution_vs_performance_{eval_type}.png"), dpi=300)
            plt.close()
    else:
        # Sort by win rate
        grouped_df = df.sort_values('Win Rate (%)', ascending=False)
        
        # Create positions for bars
        x = np.arange(len(grouped_df))
        
        # Create stacked bars for CPU and GPU memory
        plt.bar(x, grouped_df['Memory (MB)'], label='CPU Memory (MB)', color='skyblue')
        plt.bar(x, grouped_df['GPU Memory (MB)'], bottom=grouped_df['Memory (MB)'], 
                label='GPU Memory (MB)', color='orange')
        
        # Plot win rate line on secondary y-axis
        ax2 = plt.twinx()
        ax2.plot(x, grouped_df['Win Rate (%)'], 'ro-', linewidth=2, label='Win Rate (%)')
        
        # Set labels and title
        plt.xlabel('Agent', fontsize=14)
        plt.ylabel('Memory Usage (MB)', fontsize=14)
        ax2.set_ylabel('Win Rate (%)', fontsize=14, color='r')
        plt.title('Memory Usage Distribution vs. Performance', fontsize=16)
        
        # Set x-tick labels to agent names
        plt.xticks(x, grouped_df['Agent'])
        
        # Add legends
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_distribution_vs_performance.png"), dpi=300)
        plt.close()
    
    # 3. Heatmap of resource usage patterns
    plt.figure(figsize=(12, 10))
    
    # Prepare data for heatmap
    if 'Evaluation' in df.columns:
        # Create separate heatmaps for each evaluation type
        for eval_type in df['Evaluation'].unique():
            eval_df = df[df['Evaluation'] == eval_type].copy()
            
            # Create a normalized dataset for the heatmap
            heatmap_data = eval_df[['Agent', 'Win Rate (%)', 'Avg Time (ms)', 'Memory (MB)', 
                                   'GPU Memory (MB)', 'GPU Utilization (%)']].set_index('Agent')
            
            # Normalize each column
            for col in heatmap_data.columns:
                min_val = heatmap_data[col].min()
                max_val = heatmap_data[col].max()
                if max_val > min_val:
                    heatmap_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".2f")
            
            plt.title(f'Resource Usage Patterns by Agent - {eval_type}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"resource_usage_heatmap_{eval_type}.png"), dpi=300)
            plt.close()
    else:
        # Create a normalized dataset for the heatmap
        heatmap_data = df[['Agent', 'Win Rate (%)', 'Avg Time (ms)', 'Memory (MB)', 
                          'GPU Memory (MB)', 'GPU Utilization (%)']].set_index('Agent')
        
        # Normalize each column
        for col in heatmap_data.columns:
            min_val = heatmap_data[col].min()
            max_val = heatmap_data[col].max()
            if max_val > min_val:
                heatmap_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", linewidths=.5, fmt=".2f")
        
        plt.title('Resource Usage Patterns by Agent', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "resource_usage_heatmap.png"), dpi=300)
        plt.close()


def create_unified_gpu_pareto_analysis(df, output_dir):
    """
    Create a 3D Pareto analysis of both CPU and GPU resource usage
    
    Args:
        df: DataFrame with performance and resource data
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 2D Pareto chart with combined resource usage (CPU + GPU weighted)
    plt.figure(figsize=(12, 10))
    
    # Add combined resource column as a simple weighted sum
    if 'GPU Memory (MB)' in df.columns and df['GPU Memory (MB)'].max() > 0:
        df['Combined Resources (MB)'] = df['Memory (MB)'] + (0.75 * df['GPU Memory (MB)'])
    else:
        df['Combined Resources (MB)'] = df['Memory (MB)']
    
    # Create scatter plot
    if 'Evaluation' in df.columns:
        scatter = sns.scatterplot(
            data=df,
            x='Combined Resources (MB)',
            y='Win Rate (%)',
            hue='Agent',
            style='Evaluation',
            s=df['Avg Time (ms)'] * 0.5,
            alpha=0.7
        )
    else:
        scatter = sns.scatterplot(
            data=df,
            x='Combined Resources (MB)',
            y='Win Rate (%)',
            hue='Agent',
            s=df['Avg Time (ms)'] * 0.5,
            alpha=0.7
        )
    
    # Add annotations
    for i, row in df.iterrows():
        label = row['Agent']
        if 'Evaluation' in df.columns:
            label += f" ({'FE' if row['Evaluation'] == 'Fixed Episodes' else 'FR'})"
            
        plt.annotate(
            label,
            (row['Combined Resources (MB)'], row['Win Rate (%)']),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )
    
    # Add titles and labels
    plt.title('Pareto Analysis: Win Rate vs Combined Resource Usage', fontsize=16)
    plt.xlabel('Combined Resources (MB)', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend for the bubble size
    sizes = df['Avg Time (ms)'].quantile([0.25, 0.5, 0.75]).values
    labels = [f"{size:.1f} ms" for size in sizes]
    
    # Create dummy scatter points for the legend
    for size, label in zip(sizes, labels):
        plt.scatter([], [], s=size*0.5, c='gray', alpha=0.7, label=label)
    
    # Add the size legend
    plt.legend(title="Response Time", loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_combined_resources.png"), dpi=300)
    plt.close()
    
    # Try to create a 3D Pareto surface
    try:
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot in 3D
        scatter = ax.scatter(
            df['Memory (MB)'],
            df['GPU Memory (MB)'],
            df['Win Rate (%)'],
            c=df['Avg Time (ms)'],
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Response Time (ms)', fontsize=12)
        
        # Add labels and annotations
        for i, row in df.iterrows():
            label = row['Agent']
            if 'Evaluation' in df.columns:
                label += f" ({'FE' if row['Evaluation'] == 'Fixed Episodes' else 'FR'})"
                
            ax.text(
                row['Memory (MB)'],
                row['GPU Memory (MB)'],
                row['Win Rate (%)'],
                label,
                fontsize=9
            )
        
        # Add axis labels and title
        ax.set_xlabel('CPU Memory (MB)', fontsize=14)
        ax.set_ylabel('GPU Memory (MB)', fontsize=14)
        ax.set_zlabel('Win Rate (%)', fontsize=14)
        plt.title('3D Pareto Analysis: Win Rate vs CPU/GPU Memory', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3d_pareto_surface.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating 3D Pareto surface: {e}")


def run_gpu_visualization(eval_dir="./evaluations", output_dir=None):
    """
    Run all GPU-enhanced visualizations
    
    Args:
        eval_dir: Directory containing evaluation results
        output_dir: Directory to save visualizations (None for auto-generation)
        
    Returns:
        str: Path to the visualization directory
    """
    # Generate output directory if not provided
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("./visualizations", f"gpu_analysis_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {output_dir}")
    
    # Load evaluation data
    fixed_episodes_data, fixed_resource_data = load_evaluation_data(eval_dir)
    
    if not fixed_episodes_data and not fixed_resource_data:
        print("No evaluation data found. Please check the evaluation directory.")
        return None
    
    # Load CNP metrics
    cnp_metrics = load_cnp_metrics(eval_dir)
    
    # Create DataFrame for visualization
    df = create_agent_dataframe(fixed_episodes_data, fixed_resource_data, cnp_metrics)
    
    if df.empty:
        print("No data available for visualization.")
        return None
    
    # Print summary
    print("\nAgent Data Summary:")
    print(f"Number of agents: {df['Agent'].nunique()}")
    
    if 'Evaluation' in df.columns:
        print(f"Evaluation types: {', '.join(df['Evaluation'].unique())}")
        
    print("\nMetrics ranges:")
    for col in ['Win Rate (%)', 'Avg Time (ms)', 'Memory (MB)', 'GPU Memory (MB)', 'GPU Utilization (%)']:
        if col in df.columns:
            print(f"  {col}: {df[col].min():.2f} to {df[col].max():.2f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    print("1. Creating GPU Pareto charts...")
    create_gpu_pareto_chart(df, output_dir)
    
    print("2. Creating GPU enhanced radar charts...")
    create_gpu_enhanced_radar_chart(df, output_dir)
    
    print("3. Creating CNP comparison charts...")
    if 'Standard CNP' in df.columns and df['Standard CNP'].notna().any():
        create_cnp_comparison_chart(df, output_dir)
    else:
        print("  No CNP data available, skipping CNP comparison charts")
    
    print("4. Creating GPU impact visualizations...")
    create_gpu_impact_visualization(df, output_dir)
    
    print("5. Creating unified Pareto analysis...")
    create_unified_gpu_pareto_analysis(df, output_dir)
    
    print(f"\nAll visualizations have been generated successfully in: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GPU-enhanced visualizations from evaluation results")
    parser.add_argument("--eval-dir", type=str, default="./evaluations", 
                        help="Directory containing evaluation results (default: ./evaluations)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save visualizations (default: auto-generated)")
    
    args = parser.parse_args()
    
    run_gpu_visualization(args.eval_dir, args.output_dir)