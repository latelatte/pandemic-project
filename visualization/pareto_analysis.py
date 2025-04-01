import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import glob

def load_agent_data(results_dir):
    """load agent performance data from JSON file"""
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Warning: {metrics_file} not found.")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        agents_data = []
        for agent_name, metrics in data.get("agent_performance", {}).items():
            resource_usage = data.get("resource_usage", {}).get(agent_name, {})
            agent_data = {
                "Agent": agent_name,
                "Win Rate (%)": metrics.get("win_contribution", 0) * 100,
                "Avg Time (ms)": metrics.get("avg_time_ms", 0),
                "Memory (MB)": resource_usage.get("avg_memory_mb", 0),
                "CPU (%)": resource_usage.get("avg_cpu_percent", 0),
            }
            agents_data.append(agent_data)
            
        return pd.DataFrame(agents_data) if agents_data else None
        
    except Exception as e:
        print(f"error: {e}")
        return None

def aggregate_experiment_data(results_dir="./evaluations", n_latest=5, pattern="experiment_*"):
    """create a DataFrame from multiple experiment directories"""
    if os.path.isdir(results_dir) and not results_dir.endswith("logs"):
        experiment_dirs = [results_dir]
    else:
        base_dir = results_dir if results_dir.endswith("logs") else "./logs"
        experiment_dirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                               if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("experiment_")])
        experiment_dirs = experiment_dirs[-n_latest:] if len(experiment_dirs) >= n_latest else experiment_dirs
    
    all_agent_data = {}
    
    for exp_dir in experiment_dirs:
        metrics_file = glob.glob(os.path.join(results_dir, "./evaluations/*.json"))
        if not metrics_file:
            print(f"WARN: {metrics_file} not found.")
            continue
            
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                
            for agent_name, metrics in data.get("agent_performance", {}).items():
                if agent_name not in all_agent_data:
                    all_agent_data[agent_name] = {
                        "win_rates": [], "times": [], "memory": [], "cpu": []
                    }
                
                resource_usage = data.get("resource_usage", {}).get(agent_name, {})
                win_rate = metrics.get("win_contribution", 0) * 100
                avg_time = metrics.get("avg_time_ms", 0)
                memory = resource_usage.get("avg_memory_mb", 0)
                cpu = resource_usage.get("avg_cpu_percent", 0)
                
                all_agent_data[agent_name]["win_rates"].append(win_rate)
                all_agent_data[agent_name]["times"].append(avg_time)
                all_agent_data[agent_name]["memory"].append(memory)
                all_agent_data[agent_name]["cpu"].append(cpu)
                
        except Exception as e:
            print(f"error occured in processing on {metrics_file} : {e}")
    
    # convert to DataFrame
    aggregated_data = []
    for agent_name, data in all_agent_data.items():
        if not data["win_rates"]:
            continue
            
        aggregated_data.append({
            "Agent": agent_name,
            "Win Rate (%)": np.mean(data["win_rates"]),
            "Win Rate StdDev": np.std(data["win_rates"]),
            "Avg Time (ms)": np.mean(data["times"]),
            "Time StdDev": np.std(data["times"]),
            "Memory (MB)": np.mean(data["memory"]),
            "Memory StdDev": np.std(data["memory"]),
            "CPU (%)": np.mean(data["cpu"]),
            "CPU StdDev": np.std(data["cpu"]),
            "n_samples": len(data["win_rates"])
        })
    
    return pd.DataFrame(aggregated_data) if aggregated_data else None

def compute_pareto_frontier(points):
    """calculate the Pareto frontier from a list of points"""
    pareto_points = []
    for i, (x_i, y_i) in enumerate(points):
        is_dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if i != j and x_j <= x_i and y_j >= y_i and (x_j < x_i or y_j > y_i):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append((x_i, y_i))
    
    return sorted(pareto_points, key=lambda x: x[0])

def create_2d_pareto_chart(df, x_col, y_col, x_label, y_label, title, filename, output_dir, 
                          size_col=None, minimize_x=True, maximize_y=True):
    """for 2D Pareto chart"""
    plt.figure(figsize=(12, 8))
    
    if size_col:
        scatter = sns.scatterplot(
            x=x_col, y=y_col, 
            hue="Agent", size=size_col,
            sizes=(100, 400), data=df,
            alpha=0.7
        )
    else:
        scatter = sns.scatterplot(
            x=x_col, y=y_col, 
            hue="Agent", s=200,
            data=df, alpha=0.7
        )
    
    for i, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        x_err = row.get(f"{x_col.split()[0]} StdDev", 0)
        y_err = row.get(f"{y_col.split()[0]} StdDev", 0)
        
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', ecolor='gray', alpha=0.5)
    
    points = [(row[x_col], row[y_col]) for _, row in df.iterrows()]
    
    if not minimize_x:
        points = [(-x, y) for x, y in points]
    if not maximize_y:
        points = [(x, -y) for x, y in points]
    
    pareto_points = compute_pareto_frontier(points)
    if not minimize_x:
        pareto_points = [(-x, y) for x, y in pareto_points]
    if not maximize_y:
        pareto_points = [(x, -y) for x, y in pareto_points]
    
    if len(pareto_points) >= 2:
        pareto_x, pareto_y = zip(*pareto_points)
        plt.plot(pareto_x, pareto_y, 'k--', label='Pareto Frontier')
    
    for i, agent in enumerate(df["Agent"]):
        plt.annotate(
            agent,
            (df[x_col].iloc[i], df[y_col].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )
    
    max_x = plt.xlim()[1]
    min_x = plt.xlim()[0]
    max_y = plt.ylim()[1]
    min_y = plt.ylim()[0]
    
    arrow_x_pos = min_x + (max_x - min_x) * 0.05
    arrow_y_pos = min_y + (max_y - min_y) * 0.95
    
    if minimize_x:
        plt.annotate('', xy=(arrow_x_pos, arrow_y_pos), xytext=(arrow_x_pos + (max_x - min_x) * 0.1, arrow_y_pos),
                arrowprops=dict(arrowstyle='<-', color='red'))
        plt.text(arrow_x_pos + (max_x - min_x) * 0.05, arrow_y_pos - (max_y - min_y) * 0.03, 
                'Minimize', ha='center', color='red')
    else:
        plt.annotate('', xy=(arrow_x_pos + (max_x - min_x) * 0.1, arrow_y_pos), xytext=(arrow_x_pos, arrow_y_pos),
                arrowprops=dict(arrowstyle='<-', color='green'))
        plt.text(arrow_x_pos + (max_x - min_x) * 0.05, arrow_y_pos - (max_y - min_y) * 0.03, 
                'Maximize', ha='center', color='green')
    
    arrow_x_pos = min_x + (max_x - min_x) * 0.05
    arrow_y_pos = min_y + (max_y - min_y) * 0.05
    
    if maximize_y:
        plt.annotate('', xy=(arrow_x_pos, arrow_y_pos + (max_y - min_y) * 0.1), xytext=(arrow_x_pos, arrow_y_pos),
                arrowprops=dict(arrowstyle='<-', color='green'))
        plt.text(arrow_x_pos - (max_x - min_x) * 0.03, arrow_y_pos + (max_y - min_y) * 0.05, 
                'Maximize', va='center', rotation=90, color='green')
    else:
        plt.annotate('', xy=(arrow_x_pos, arrow_y_pos), xytext=(arrow_x_pos, arrow_y_pos + (max_y - min_y) * 0.1),
                arrowprops=dict(arrowstyle='<-', color='red'))
        plt.text(arrow_x_pos - (max_x - min_x) * 0.03, arrow_y_pos + (max_y - min_y) * 0.05, 
                'Minimize', va='center', rotation=90, color='red')
    
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Agent", fontsize=10, title_fontsize=12)
    
    if len(pareto_points) >= 2:
        pareto_x, pareto_y = zip(*pareto_points)
        if minimize_x and maximize_y:
            plt.fill_between(pareto_x, pareto_y, [max_y] * len(pareto_x), 
                            alpha=0.1, color='green', label='_ideal_region')
            
            plt.annotate('Desireble', xy=(min_x + (max_x - min_x) * 0.2, max_y - (max_y - min_y) * 0.2),
                        xytext=(min_x + (max_x - min_x) * 0.3, max_y - (max_y - min_y) * 0.3),
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.5),
                        color='green', alpha=0.7, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def create_2d_pareto_analysis(df, output_dir):
    """create 2D Pareto analysis charts"""
    os.makedirs(output_dir, exist_ok=True)

    create_2d_pareto_chart(
        df, 
        "Avg Time (ms)", "Win Rate (%)",
        "Average Response Time (ms) - The smaller the better", "Win Rate (%) - The higher the better",
        "Performance vs Win Rate Trade-off Analysis",
        "pareto_time_winrate.png",
        output_dir,
        "Memory (MB)"
    )
    
    create_2d_pareto_chart(
        df, 
        "Memory (MB)", "Win Rate (%)",
        "Memory Usage (MB) - The smaller the better", "Win Rate (%) - The higher the better",
        "Memory vs Win Rate Trade-off Analysis",
        "pareto_memory_winrate.png",
        output_dir,
        "Avg Time (ms)"
    )

def create_3d_visualization(df, output_dir):
    """for 3D visualization of performance metrics"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    markers = ['o', '^', 's', 'D', '*']
    colormap = plt.cm.viridis
    colors = [colormap(i) for i in np.linspace(0, 1, len(df['Agent'].unique()))]
    
    for i, agent in enumerate(df['Agent'].unique()):
        agent_data = df[df['Agent'] == agent]
        ax.scatter(
            agent_data['Avg Time (ms)'].values,
            agent_data['Memory (MB)'].values,
            agent_data['Win Rate (%)'].values,
            label=agent,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            s=150
        )
        
        for j, row in agent_data.iterrows():
            x = row['Avg Time (ms)']
            y = row['Memory (MB)']
            z = row['Win Rate (%)']
            dx = row.get('Time StdDev', 0)
            dy = row.get('Memory StdDev', 0)
            dz = row.get('Win Rate StdDev', 0)
            
            # X-axis error bar
            ax.plot([x-dx, x+dx], [y, y], [z, z], color='gray', alpha=0.3)
            
            # Y-axis error bar
            ax.plot([x, x], [y-dy, y+dy], [z, z], color='gray', alpha=0.3)
            
            # Z-axis error bar
            ax.plot([x, x], [y, y], [z-dz, z+dz], color='gray', alpha=0.3)
    
    ax.set_xlabel('Average Response Time (ms)', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_zlabel('Win Rate (%)', fontsize=12)
    
    max_x = max(df['Avg Time (ms)'].max(), 1)
    max_y = max(df['Memory (MB)'].max(), 1)
    max_z = max(df['Win Rate (%)'].max(), 1)
    
    ax.quiver(max_x * 0.9, 0, 0, -max_x * 0.2, 0, 0, color='r', arrow_length_ratio=0.1, label='Minimize')
    ax.text(max_x * 0.8, 0, -max_z * 0.1, "Time ↓", color='r')
    
    ax.quiver(0, max_y * 0.9, 0, 0, -max_y * 0.2, 0, color='r', arrow_length_ratio=0.1)
    ax.text(0, max_y * 0.8, -max_z * 0.1, "Memory ↓", color='r')
    
    ax.quiver(0, 0, max_z * 0.7, 0, 0, max_z * 0.2, color='g', arrow_length_ratio=0.1)
    ax.text(0, -max_y * 0.1, max_z * 0.8, "Win Rate ↑", color='g')
    
    ax.view_init(elev=30, azim=45)
    plt.title("3D Performance Analysis with Error Bars", fontsize=16)
    plt.legend(title="Agent", fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3d_performance_analysis.png"), dpi=300)
    plt.close()

def create_radar_chart(df, output_dir):
    """visualize performance metrics using radar chart"""
    df_norm = df.copy()

    maximize_metrics = ["Win Rate (%)"]
    minimize_metrics = ["Avg Time (ms)", "Memory (MB)"]
    
    # normalize metrics
    for col in maximize_metrics + minimize_metrics:
        max_val = df[col].max()
        min_val = df[col].min()
        range_val = max_val - min_val
        
        if range_val > 0:
            if col in minimize_metrics:
                df_norm[col] = 1 - ((df[col] - min_val) / range_val)
            else:
                df_norm[col] = (df[col] - min_val) / range_val
        else:
            df_norm[col] = 0.5  # if no variation, set to 0.5
    
    categories = ['Win Rate', 'Time Efficiency', 'Memory Efficiency']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for i, agent in enumerate(df_norm['Agent'].unique()):
        agent_data = df_norm[df_norm['Agent'] == agent]
        values = agent_data[["Win Rate (%)", "Avg Time (ms)", "Memory (MB)"]].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='-', label=agent)
        ax.fill(angles, values, alpha=0.1)
    
    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1)
    
    plt.annotate('The higher the win rate, the better', xy=(0, 0.9), xytext=(0.2, 1.1), 
                textcoords='axes fraction', ha='center',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.annotate('The less proccessing time, the better', xy=(2*np.pi/3, 0.9), xytext=(0.8, 1.1), 
                textcoords='axes fraction', ha='center',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.annotate('The less memory consumption, the better', xy=(4*np.pi/3, 0.9), xytext=(0.5, 1.2), 
                textcoords='axes fraction', ha='center',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), title="Agent")
    plt.title("Multi-dimensional Agent Comparison", size=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_comparison.png"), dpi=300)
    plt.close()

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
            row['Agent'], 
            (row['GPU Memory (MB)'], row['Win Rate (%)']),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10
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
            row['Agent'], 
            (row['GPU Utilization (%)'], row['Win Rate (%)']),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10
        )
    
    plt.savefig(os.path.join(output_dir, "gpu_utilization_vs_winrate.png"), dpi=300)
    plt.close()
    
    # 3. 3D visualization with CPU Memory, GPU Memory, and Win Rate
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
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
            ax.text(
                row['Memory (MB)'],
                row['GPU Memory (MB)'],
                row['Win Rate (%)'],
                row['Agent'],
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
    
    # Normalize metrics for radar chart
    df_norm = df.copy()
    
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
        max_val = df[col].max()
        min_val = df[col].min()
        range_val = max_val - min_val
        
        if range_val > 0:
            if col in maximize_metrics:
                # For metrics we want to maximize, higher values should be closer to 1
                df_norm[col] = (df[col] - min_val) / range_val
            else:
                # For metrics we want to minimize, lower values should be closer to 1
                df_norm[col] = 1 - ((df[col] - min_val) / range_val)
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
    
    # Draw one line per agent with different colors
    for i, agent in enumerate(df['Agent'].unique()):
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
    
    # Create grouped bar chart for CNP comparison
    plt.figure(figsize=(14, 8))
    
    # Set width of bars
    bar_width = 0.25
    index = np.arange(len(df['Agent']))
    
    # Create bars
    plt.bar(index, df['Standard CNP'], bar_width, label='Standard CNP', color='skyblue')
    plt.bar(index + bar_width, df['CPU-only CNP'], bar_width, label='CPU-only CNP', color='lightgreen')
    plt.bar(index + 2 * bar_width, df['GPU-focused CNP'], bar_width, label='GPU-focused CNP', color='salmon')
    
    # Add labels and title
    plt.xlabel('Agent', fontsize=14)
    plt.ylabel('CNP Value', fontsize=14)
    plt.title('Comparison of CNP Calculation Methods', fontsize=16)
    plt.xticks(index + bar_width, df['Agent'])
    plt.legend()
    
    # Add value labels on top of bars
    for i, value in enumerate(df['Standard CNP']):
        plt.text(i, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
    
    for i, value in enumerate(df['CPU-only CNP']):
        plt.text(i + bar_width, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
    
    for i, value in enumerate(df['GPU-focused CNP']):
        plt.text(i + 2 * bar_width, value + 0.1, f"{value:.2f}", ha='center', va='bottom', fontsize=9)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cnp_comparison.png"), dpi=300)
    plt.close()
    
    # Create pie charts showing resource distribution for each agent
    for i, row in df.iterrows():
        agent = row['Agent']
        
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
        plt.title(f'Resource Distribution for {agent}', size=16)
        
        # Save the figure
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.savefig(os.path.join(output_dir, f"{agent}_resource_distribution.png"), dpi=300)
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
        plt.annotate(
            row['Agent'], 
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
    
    # Sort by win rate
    df_sorted = df.sort_values('Win Rate (%)', ascending=False)
    
    # Create positions for bars
    x = np.arange(len(df_sorted))
    
    # Create stacked bars for CPU and GPU memory
    plt.bar(x, df_sorted['Memory (MB)'], label='CPU Memory (MB)', color='skyblue')
    plt.bar(x, df_sorted['GPU Memory (MB)'], bottom=df_sorted['Memory (MB)'], 
            label='GPU Memory (MB)', color='orange')
    
    # Plot win rate line on secondary y-axis
    ax2 = plt.twinx()
    ax2.plot(x, df_sorted['Win Rate (%)'], 'ro-', linewidth=2, label='Win Rate (%)')
    
    # Set labels and title
    plt.xlabel('Agent', fontsize=14)
    plt.ylabel('Memory Usage (MB)', fontsize=14)
    ax2.set_ylabel('Win Rate (%)', fontsize=14, color='r')
    plt.title('Memory Usage Distribution vs. Performance', fontsize=16)
    
    # Set x-tick labels to agent names
    plt.xticks(x, df_sorted['Agent'])
    
    # Add legends
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_distribution_vs_performance.png"), dpi=300)
    plt.close()
    
    # 3. Heatmap of resource usage patterns
    plt.figure(figsize=(12, 10))
    
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


# Example usage of these visualization functions
def create_all_gpu_visualizations(data, output_dir="./gpu_visualizations"):
    """
    Create all GPU-enhanced visualizations
    
    Args:
        data: Dictionary with agent performance data
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for visualization
    agents = []
    for agent_name, metrics in data.items():
        # Get the latest win rate and resource metrics
        win_rate = metrics.get("fixed_resource_win_rate", 0)
        avg_time_ms = metrics.get("agent_performance", {}).get(agent_name, {}).get("avg_time_ms", 0)
        memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_memory_mb", 0)
        gpu_memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_gpu_memory_mb", 0)
        gpu_percent = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_gpu_percent", 0)
        
        # Get CNP calculations
        standard_cnp = metrics.get("fixed_resource_cnp", 0)
        cpu_only_cnp = metrics.get("fixed_resource_cpu_only_cnp", 0)
        gpu_focused_cnp = metrics.get("fixed_resource_gpu_focused_cnp", 0)
        
        # Get resource distribution
        resource_dist = metrics.get("fixed_resource_resource_distribution", {})
        cpu_percent = resource_dist.get("cpu_percent", 50)
        gpu_percent = resource_dist.get("gpu_percent", 50)
        
        agents.append({
            "Agent": agent_name,
            "Win Rate (%)": win_rate,
            "Avg Time (ms)": avg_time_ms,
            "Memory (MB)": memory_mb,
            "GPU Memory (MB)": gpu_memory_mb,
            "GPU Utilization (%)": gpu_percent,
            "Standard CNP": standard_cnp,
            "CPU-only CNP": cpu_only_cnp,
            "GPU-focused CNP": gpu_focused_cnp,
            "CPU Resource %": cpu_percent,
            "GPU Resource %": gpu_percent
        })
    
    df = pd.DataFrame(agents)
    
    # Create visualizations
    create_gpu_pareto_chart(df, output_dir)
    create_gpu_enhanced_radar_chart(df, output_dir)
    create_cnp_comparison_chart(df, output_dir)
    create_gpu_impact_visualization(df, output_dir)
    
    print(f"All GPU-enhanced visualizations created in: {output_dir}")
    return df

def run_pareto_analysis(evaluation_dir, output_dir=None, n_latest=5):
    report_files = glob.glob(os.path.join(evaluation_dir, "integrated_evaluation_report_*.json"))
    if not report_files:
        print("No integrated evaluation report found.")
        return False
    
    latest_report = max(report_files, key=os.path.getctime)
    
    adapted_dir = os.path.join(evaluation_dir, "adapted_data")
    metrics_path = convert_integrated_report_for_pareto(latest_report, adapted_dir)
    
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    agent_data = []
    for agent_name, metrics in metrics_data.get("agent_performance", {}).items():
        resource_usage = metrics_data.get("resource_usage", {}).get(agent_name, {})
        agent_data.append({
            "Agent": agent_name,
            "Win Rate (%)": metrics.get("win_contribution", 0) * 100,
            "Avg Time (ms)": metrics.get("avg_time_ms", 0),
            "Memory (MB)": resource_usage.get("avg_memory_mb", 0),
            "CPU (%)": resource_usage.get("avg_cpu_percent", 0),
            "n_samples": 1 
        })
    
    df = pd.DataFrame(agent_data)
    
    if output_dir is None:
        output_dir = os.path.join(evaluation_dir, "analysis")

    create_2d_pareto_analysis(df, output_dir)
    create_3d_visualization(df, output_dir)
    create_radar_chart(df, output_dir)
    # create_cnp_comparison_chart(df, output_dir)
    # create_gpu_impact_visualization(df, output_dir)
    # create_gpu_pareto_chart(df, output_dir)
    # create_gpu_enhanced_radar_chart(df, output_dir)
    print(f"Pareto analysis completed. Visualizations saved to: {output_dir}")
    return True


def convert_integrated_report_for_pareto(integrated_report_path, output_dir="./adaptedData"):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(integrated_report_path, 'r') as f:
        report = json.load(f)
    
    fixed_episodes = report.get("fixed_episodes_results", {})
    fixed_resources = report.get("fixed_resource_results", {})
    
    metrics = {"agent_performance": {}, "resource_usage": {}}
    
    for agent_name, data in fixed_episodes.items():
        metrics["agent_performance"][agent_name] = data.get("agent_performance", {}).get(agent_name, {})
        metrics["resource_usage"][agent_name] = data.get("resource_usage", {}).get(agent_name, {})
        
    for agent_name, data in fixed_resources.items():
        resource_agent_name = f"{agent_name} (Fixed Resource)"
        metrics["agent_performance"][resource_agent_name] = data.get("agent_performance", {}).get(agent_name, {})
        metrics["resource_usage"][resource_agent_name] = data.get("resource_usage", {}).get(agent_name, {})
        
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics_path

def run_adapted_pareto_analysis(evaluation_dir="./evaluations"):
    report_files = glob.glob(os.path.join(evaluation_dir, "integrated_evaluation_report_*.json"))
    if not report_files:
        print("No integrated evaluation report found.")
        return False
    
    latest_report = max(report_files, key=os.path.getctime)
    
    adapted_dir = os.path.join(evaluation_dir, "adapted_data")
    metrics_path = convert_integrated_report_for_pareto(latest_report, adapted_dir)
    
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    agent_data = []
    for agent_name, metrics in metrics_data.get("agent_performance", {}).items():
        resource_usage = metrics_data.get("resource_usage", {}).get(agent_name, {})
        agent_data.append({
            "Agent": agent_name,
            "Win Rate (%)": metrics.get("win_contribution", 0) * 100,
            "Avg Time (ms)": metrics.get("avg_time_ms", 0),
            "Memory (MB)": resource_usage.get("avg_memory_mb", 0),
            "CPU (%)": resource_usage.get("avg_cpu_percent", 0),
            "n_samples": 1 
        })
    
    df = pd.DataFrame(agent_data)
    
    output_dir = os.path.join(evaluation_dir, "pareto_plots")
    create_2d_pareto_analysis(df, output_dir)
    create_3d_visualization(df, output_dir)
    create_radar_chart(df, output_dir)
    # create_cnp_comparison_chart(df, output_dir)
    # create_gpu_impact_visualization(df, output_dir)
    # create_gpu_pareto_chart(df, output_dir)
    # create_gpu_enhanced_radar_chart(df, output_dir)
    return True


if __name__ == "__main__":
    import sys
    
    def find_latest_experiment():
        log_dirs = sorted([d for d in os.listdir("./logs") if d.startswith("experiment_")])
        if not log_dirs:
            return None
        return os.path.join("./logs", log_dirs[-1])
    
    import argparse
    parser = argparse.ArgumentParser(description='pareto analysis')
    parser.add_argument('--dir', type=str, default="./evaluations", help='target experiment directory (デフォルト: ../logs)')
    parser.add_argument('--n_latest', type=int, default=5, help='number of latest experiments to analyze (デフォルト: 5)')
    parser.add_argument('--output', type=str, default="./plots", help='output directory for plots (デフォルト: ./plots)')
    
    args = parser.parse_args()
    results_dir = args.dir if args.dir else find_latest_experiment()
    
    if not results_dir:
        print("not found experiment directory")
        sys.exit(1)
    
    success = run_pareto_analysis(results_dir, args.output, args.n_latest)