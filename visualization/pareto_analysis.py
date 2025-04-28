import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import glob

TITLE_SIZE = 22 + 2
LABEL_SIZE = 20 + 2
TICK_SIZE = 18 + 2
LEGEND_SIZE = 18 + 2
ANNOTATION_SIZE = 18 + 2
SMALL_ANNOTATION_SIZE = 16 + 2
FIG_TEXT_SIZE = 13 + 2

def find_project_root():
    """
    プロジェクトルートディレクトリを自動検出する
    
    Returns:
        str: プロジェクトルートのパス
    """
    current_path = os.path.abspath(__file__)
    
    current_dir = os.path.dirname(current_path)
    
    parent_dir = os.path.dirname(current_dir)
    
    if os.path.exists(os.path.join(parent_dir, "evaluations")):
        print(f"Found project root: {parent_dir}")
        return parent_dir
    

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
                "Win Rate (%)": metrics.get("win_rate", 0) * 100,
                "Avg Time (ms)": metrics.get("avg_time_ms", 0),
                "Memory (MB)": resource_usage.get("avg_memory_mb", 0),
                "CPU (%)": resource_usage.get("avg_cpu_percent", 0),
            }
            agents_data.append(agent_data)
            
        return pd.DataFrame(agents_data) if agents_data else None
        
    except Exception as e:
        print(f"error: {e}")
        return None

def aggregate_experiment_data(results_dir="./evaluations", n_latest=5, pattern="experiment_*", use_multiple_experiments=False):
    """create a DataFrame from multiple experiment directories"""
    if results_dir.startswith('./') or results_dir.startswith('../'):
        project_root = find_project_root()
        if results_dir == './evaluations':
            results_dir = os.path.join(project_root, 'evaluations')
            print(f"Using project root evaluations directory: {results_dir}")
    
    results_dir = os.path.abspath(results_dir)
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        return None

    if os.path.isdir(results_dir) and not results_dir.endswith("logs"):
        if use_multiple_experiments:
            report_files = []
            
            integrated_reports = glob.glob(os.path.join(results_dir, "integrated_evaluation_report_*.json"))
            evaluation_reports = glob.glob(os.path.join(results_dir, "*evaluation_report_*.json"))
            
            evaluation_reports = [f for f in evaluation_reports if "integrated" not in os.path.basename(f).lower()]
            
            report_files.extend(integrated_reports)
            report_files.extend(evaluation_reports)
            
            for root, dirs, files in os.walk(results_dir):
                if root != results_dir:
                    for file in files:
                        if file.endswith(".json") and "evaluation_report" in file:
                            report_files.append(os.path.join(root, file))
            
            report_files = list(set(report_files))
            
            print(f"Found {len(report_files)} evaluation reports:")
            for report_file in report_files[:5]:
                print(f"  - {os.path.basename(report_file)}")
            if len(report_files) > 5:
                print(f"  ... and {len(report_files) - 5} more")
            
            experiment_dirs = [results_dir]
        else:
            experiment_dirs = [results_dir]
    else:
        base_dir = results_dir if results_dir.endswith("logs") else "./logs"
        experiment_dirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                               if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("experiment_")])
        experiment_dirs = experiment_dirs[-n_latest:] if len(experiment_dirs) >= n_latest else experiment_dirs
    
    all_agent_data = {}
    
    if use_multiple_experiments and 'report_files' in locals():
        for report_file in report_files:
            try:
                print(f"Processing report: {report_file}")
                
                with open(report_file, 'r') as f:
                    data = json.load(f)
                
                for eval_type in ["fixed_episodes_results", "fixed_resource_results"]:
                    for agent_name, metrics in data.get(eval_type, {}).items():
                        eval_display_name = "Fixed Episodes" if "episodes" in eval_type else "Fixed Resource"
                        qualified_agent_name = f"{agent_name} ({eval_display_name})"
                        
                        if qualified_agent_name not in all_agent_data:
                            all_agent_data[qualified_agent_name] = {
                                "win_rates": [], "times": [], "memory": [], "cpu": []
                            }
                        
                        win_rate = metrics.get("win_rate", 0) * 100
                        
                        agent_perf = metrics.get("agent_performance", {}).get(agent_name, {})
                        avg_time = agent_perf.get("avg_time_ms", 0)
                        
                        resource_usage = metrics.get("resource_usage", {}).get(agent_name, {})
                        memory = resource_usage.get("avg_memory_mb", 0)
                        cpu = resource_usage.get("avg_cpu_percent", 0)
                        
                        print(f"Agent: {qualified_agent_name}, Win Rate: {win_rate:.2f}%, Memory: {memory}, Source: {os.path.basename(report_file)}")
                        
                        all_agent_data[qualified_agent_name]["win_rates"].append(win_rate)
                        all_agent_data[qualified_agent_name]["times"].append(avg_time)
                        all_agent_data[qualified_agent_name]["memory"].append(memory)
                        all_agent_data[qualified_agent_name]["cpu"].append(cpu)
                
            except Exception as e:
                print(f"Error occurred in processing {report_file}: {e}")
    else:
        for exp_dir in experiment_dirs:
            metrics_files = glob.glob(os.path.join(exp_dir, "*.json"))
            if not metrics_files:
                metrics_files = glob.glob(os.path.join(exp_dir, "evaluations", "*.json"))
                
            if not metrics_files:
                print(f"WARN: No JSON metrics files found in {exp_dir}")
                continue
                
            metrics_file = max(metrics_files, key=os.path.getctime)
            print(f"Processing metrics file: {metrics_file}")
                
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    
                for agent_name, metrics in data.get("agent_performance", {}).items():
                    if agent_name not in all_agent_data:
                        all_agent_data[agent_name] = {
                            "win_rates": [], "times": [], "memory": [], "cpu": []
                        }
                    
                    resource_usage = data.get("resource_usage", {}).get(agent_name, {})
                    win_rate = metrics.get("win_rate", 0) * 100
                    avg_time = metrics.get("avg_time_ms", 0)
                    memory = resource_usage.get("avg_memory_mb", 0)
                    cpu = resource_usage.get("avg_cpu_percent", 0)
                    
                    print(f"Agent: {agent_name}, Memory: {memory}, Source: {metrics_file}")
                    
                    all_agent_data[agent_name]["win_rates"].append(win_rate)
                    all_agent_data[agent_name]["times"].append(avg_time)
                    all_agent_data[agent_name]["memory"].append(memory)
                    all_agent_data[agent_name]["cpu"].append(cpu)
                    
            except Exception as e:
                print(f"Error occurred in processing {metrics_file}: {e}")
    
    # convert to DataFrame with statistical measures
    aggregated_data = []
    for agent_name, data in all_agent_data.items():
        if not data["win_rates"]:
            continue
            
        print(f"Agent {agent_name} data from all experiments:")
        for i, win_rate in enumerate(data["win_rates"]):
            print(f"  Experiment {i+1}: Win Rate = {win_rate:.2f}%, Time = {data['times'][i]:.2f}ms, Memory = {data['memory'][i]:.2f}MB")
            
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
    
    df = pd.DataFrame(aggregated_data) if aggregated_data else None
    
    if df is not None:
        print("\nAggregate Statistics:")
        for _, row in df.iterrows():
            agent = row["Agent"]
            win_rate = row["Win Rate (%)"]
            win_std = row["Win Rate StdDev"]
            memory = row["Memory (MB)"]
            memory_std = row["Memory StdDev"]
            n = row["n_samples"]
            
            print(f"{agent}: Win Rate = {win_rate:.2f}% ±{win_std:.2f} (n={n}), Memory = {memory:.2f}MB ±{memory_std:.2f}")
        
    return df


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
                          size_col=None, minimize_x=True, maximize_y=True, show_size_values=False):
    """for 2D Pareto chart"""
    plt.figure(figsize=(12, 8))
    
    df_cleaned = df.copy()
    
    if 'Evaluation' not in df_cleaned.columns:
        df_cleaned['Evaluation'] = df_cleaned['Agent'].apply(
            lambda x: x.split('(')[-1].replace(')', '') if '(' in x else 'Unknown'
        )
        df_cleaned['Agent'] = df_cleaned['Agent'].apply(
            lambda x: x.split(' (')[0] if ' (' in x else x
        )
    
    df_cleaned['AgentSimple'] = df_cleaned['Agent'].str.replace('Agent', '')
    
    colors = {
        'MCTS': 'purple',
        'EA': 'green',
        'MARL': 'orange'
    }
    
    markers = {
        'Fixed Episodes': 'o',
        'Fixed Resource': '^'
    }

    for agent_group in df_cleaned['AgentSimple'].unique():
        for eval_type in df_cleaned['Evaluation'].unique():
            subset = df_cleaned[(df_cleaned['AgentSimple'] == agent_group) & 
                              (df_cleaned['Evaluation'] == eval_type)]
            if not subset.empty:
                color = colors.get(agent_group, 'blue')
                marker = markers.get(eval_type, 'o')
                
                plt.scatter(
                    subset[x_col], 
                    subset[y_col],
                    color=color,
                    marker=marker,
                    s=200,
                    label=f"{agent_group} ({eval_type})",
                    alpha=0.7
                )
    
    for i, row in df_cleaned.iterrows():
        x = row[x_col]
        y = row[y_col]
        x_err = row.get(f"{x_col.split()[0]} StdDev", 0)
        y_err = row.get(f"{y_col.split()[0]} StdDev", 0)
        
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', ecolor='gray', alpha=0.5)
    
    points = [(row[x_col], row[y_col]) for _, row in df_cleaned.iterrows()]
    
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
    
    annotation_offsets = {
        "EA (Fixed Episodes)": (25, 15), 
        "EA (Fixed Resource)": (-45, -25),
        "MCTS (Fixed Episodes)": (-45, -35), 
        "MCTS (Fixed Resource)": (-45, 15),
        "MARL (Fixed Episodes)": (15, 15), 
        "MARL (Fixed Resource)": (15, -15),
    }
    
    for i, row in df_cleaned.iterrows():
        x = row[x_col]
        y = row[y_col]
        
        label = f"{row['AgentSimple']} ({row['Evaluation']})"
        
        x_offset, y_offset = 5, -15
        
        if label in annotation_offsets:
            x_offset, y_offset = annotation_offsets[label]
        
        plt.annotate(
            label,
            (x, y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=ANNOTATION_SIZE,
            alpha=0.9
        )
    
    max_x = plt.xlim()[1]
    min_x = plt.xlim()[0]
    max_y = plt.ylim()[1]
    min_y = plt.ylim()[0]
    
    plt.legend(fontsize=LEGEND_SIZE, loc='best', framealpha=0.9)
    
    plt.title(title, fontsize=TITLE_SIZE)
    plt.xlabel(x_label, fontsize=LABEL_SIZE)
    plt.ylabel(y_label, fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if len(pareto_points) >= 2:
        pareto_x, pareto_y = zip(*pareto_points)
        if minimize_x and maximize_y:
            plt.fill_between(pareto_x, pareto_y, [max_y] * len(pareto_x), 
                            alpha=0.1, color='green', label='_ideal_region')
            
            plt.annotate('Desirable', xy=(min_x + (max_x - min_x) * 0.2, max_y - (max_y - min_y) * 0.2),
                        xytext=(min_x + (max_x - min_x) * 0.3, max_y - (max_y - min_y) * 0.3),
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.5),
                        color='green', alpha=0.7, fontsize=ANNOTATION_SIZE)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def create_2d_pareto_analysis(df, output_dir, show_size_values=False):
    """create 2D Pareto analysis charts"""
    os.makedirs(output_dir, exist_ok=True)

    create_2d_pareto_chart(
        df, 
        "Avg Time (ms)", "Win Rate (%)",
        "Average Response Time (ms)", "Win Rate (%)",
        "Time vs Win Rate Trade-off Analysis",
        "pareto_time_winrate.png",
        output_dir,
        "Memory (MB)",
        show_size_values=show_size_values
    )
    
    create_2d_pareto_chart(
        df, 
        "Memory (MB)", "Win Rate (%)",
        "Memory Usage (MB)", "Win Rate (%)",
        "Memory vs Win Rate Trade-off Analysis",
        "pareto_memory_winrate.png",
        output_dir,
        "Avg Time (ms)",
        show_size_values=show_size_values
    )

def create_3d_visualization(df, output_dir):
    """for 3D visualization of performance metrics"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    df_3d = df.copy()
    if 'Evaluation' not in df_3d.columns:
        df_3d['Evaluation'] = df_3d['Agent'].apply(
            lambda x: x.split('(')[-1].replace(')', '') if '(' in x else 'Unknown'
        )
        df_3d['Agent'] = df_3d['Agent'].apply(
            lambda x: x.split(' (')[0] if ' (' in x else x
        )
    
    df_3d['Agent'] = df_3d['Agent'].str.replace('Agent', '')
    
    markers = {
        'Fixed Episodes': 'o',
        'Fixed Resource': '^'
    }
    
    colors = {
        'MCTS': 'purple',
        'EA': 'green',
        'MARL': 'orange'
    }
    
    legend_handles = []
    legend_labels = []
    
    for agent_name in sorted(df_3d['Agent'].unique()):
        color = colors.get(agent_name, 'blue')
        handle = ax.scatter([], [], [], color=color, s=150, label=agent_name)
        legend_handles.append(handle)
        legend_labels.append(agent_name)
    
    for eval_type, marker in markers.items():
        handle = ax.scatter([], [], [], color='gray', marker=marker, s=150, label=eval_type)
        legend_handles.append(handle)
        legend_labels.append(eval_type)

    for _, row in df_3d.iterrows():
        agent = row['Agent']
        eval_type = row['Evaluation']
        x = row['Avg Time (ms)']
        y = row['Memory (MB)']
        z = row['Win Rate (%)']
        
        color = colors.get(agent, 'blue')
        marker = markers.get(eval_type, 'o')
        
        ax.scatter(
            x, y, z,
            color=color,
            marker=marker,
            s=150
        )
        
        dx = row.get('Time StdDev', 0)
        dy = row.get('Memory StdDev', 0)
        dz = row.get('Win Rate StdDev', 0)
        
        # error bars
        ax.plot([x-dx, x+dx], [y, y], [z, z], color='gray', alpha=0.3)
        ax.plot([x, x], [y-dy, y+dy], [z, z], color='gray', alpha=0.3)
        ax.plot([x, x], [y, y], [z-dz, z+dz], color='gray', alpha=0.3)

    ax.set_xlabel('Average Response Time (ms)', fontsize=LABEL_SIZE - 6)
    ax.set_ylabel('Memory Usage (MB)', fontsize=LABEL_SIZE - 6)
    ax.set_zlabel('Win Rate (%)', fontsize=LABEL_SIZE - 6)
    
    max_x = max(df['Avg Time (ms)'].max(), 1)
    max_y = max(df['Memory (MB)'].max(), 1)
    max_z = max(df['Win Rate (%)'].max(), 1)
    
    ax.quiver(max_x * 0.9, 0, 0, -max_x * 0.2, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.text(max_x * 0.8, 0, -max_z * 0.1, "Time ↓", color='r')
    
    ax.quiver(0, max_y * 0.9, 0, 0, -max_y * 0.2, 0, color='r', arrow_length_ratio=0.1)
    ax.text(0, max_y * 0.8, -max_z * 0.1, "Memory ↓", color='r')
    
    ax.quiver(0, 0, max_z * 0.7, 0, 0, max_z * 0.2, color='g', arrow_length_ratio=0.1)
    ax.text(0, -max_y * 0.1, max_z * 0.8, "Win Rate ↑", color='g')
    
    ax.view_init(elev=25, azim=40)
    ax.set_xlim(0, max_x * 1.2)
    ax.set_ylim(0, max_y * 1.2)
    ax.set_zlim(0, max_z * 1.2)
    
    plt.title("3D Performance Analysis with Error Bars", fontsize=TITLE_SIZE)
    
    ax.legend(legend_handles, legend_labels, fontsize=SMALL_ANNOTATION_SIZE - 2, framealpha=0.9, loc='upper right', bbox_to_anchor=(1.4, 1.02),)
    
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=TICK_SIZE - 4)
    plt.subplots_adjust(left=0.030, right=0.985, top=0.923, bottom=0.120)
    plt.savefig(os.path.join(output_dir, "3d_performance_analysis.png"), dpi=300)
    plt.close()

def create_radar_chart(df, output_dir):
    """visualize performance metrics using radar chart with improved readability"""
    df_radar = df.copy()
    if 'Evaluation' not in df_radar.columns:
        df_radar['Evaluation'] = df_radar['Agent'].apply(
            lambda x: x.split('(')[-1].replace(')', '') if '(' in x else 'Unknown'
        )
        df_radar['Agent'] = df_radar['Agent'].apply(
            lambda x: x.split(' (')[0].replace('Agent', '') if ' (' in x else x.replace('Agent', '')
        )
    
    df_radar['Legend'] = df_radar.apply(
        lambda row: f"{row['Agent']} ({row['Evaluation']})", axis=1
    )
    
    categories = ['Win Rate', 'Time Efficiency', 'Memory Efficiency']
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, polar=True)
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    linestyles = {
        'Fixed Episodes': '-',
        'Fixed Resource': '--'
    }
    
    markers = {
        'MCTS': 'o',
        'EA': 's',
        'MARL': '^'
    }
    
    for i, legend_name in enumerate(df_radar['Legend'].unique()):
        agent_data = df_radar[df_radar['Legend'] == legend_name]
        
        agent_name = agent_data['Agent'].iloc[0]
        eval_type = agent_data['Evaluation'].iloc[0]
        
        win_rate = agent_data['Win Rate (%)'].iloc[0] / 100.0

        avg_time = agent_data['Avg Time (ms)'].iloc[0]
        time_efficiency = 1.0 / (1.0 + avg_time / 100.0)
        
        memory = agent_data['Memory (MB)'].iloc[0]
        memory_efficiency = 1.0 / (1.0 + memory / 100.0)

        values = [win_rate, time_efficiency, memory_efficiency]
        values = [max(0.0, min(1.0, v)) for v in values]
        values += values[:1]

        linestyle = linestyles.get(eval_type, '-')
        marker = markers.get(agent_name, 'o')
        
        ax.plot(angles, values, linewidth=2, linestyle=linestyle, label=legend_name,
               marker=marker, markersize=8)
        ax.fill(angles, values, alpha=0.1)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    clean_labels = [label.replace('Agent', '') for label in labels]

    plt.xticks(angles[:-1], categories, fontsize=TICK_SIZE + 2)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=16)
    plt.ylim(0, 1)

    # plt.figtext(0.72, 0.47, "Higher is better", ha='center', va='center', fontsize=14)
    
    # plt.figtext(0.45, 0.95, "Higher is better", ha='center', va='center', fontsize=14)
    
    # plt.figtext(0.45, 0.10, "Higher is better", ha='center', va='center', fontsize=14)
    

    plt.legend(handles, clean_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, 
              fontsize=LEGEND_SIZE, framealpha=0.9)
    
    plt.title("Multi-dimensional Agent Comparison", fontsize=TITLE_SIZE, pad=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(left=0.090, right=0.985, top=0.923, bottom=0.120)
    plt.savefig(os.path.join(output_dir, "radar_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def convert_integrated_report_for_pareto(integrated_report_path, output_dir="./adaptedData"):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(integrated_report_path, 'r') as f:
        report = json.load(f)

    
    fixed_episodes = report.get("fixed_episodes_results", {})
    fixed_resources = report.get("fixed_resource_results", {})
    
    if fixed_episodes and list(fixed_episodes.keys()):
        first_agent = list(fixed_episodes.keys())[0]
        print(f"Sample agent data structure: {list(fixed_episodes[first_agent].keys())}")
        print(f"Win rate key exists: {'win_rate' in fixed_episodes[first_agent]}")
    
    metrics = {"agent_performance": {}, "resource_usage": {}}
    
    for agent_name, data in fixed_episodes.items():
        episodes_agent_name = f"{agent_name} (Fixed Episodes)"
        metrics["agent_performance"][episodes_agent_name] = {"win_rate": data.get("win_rate", 0)}
        
        agent_perf = data.get("agent_performance", {}).get(agent_name, {})
        if agent_perf:
            metrics["agent_performance"][episodes_agent_name].update(agent_perf)
        
        resource_data = data.get("resource_usage", {}).get(agent_name, {})
        memory_value = resource_data.get("avg_memory_mb", 0)
        print(f"Fixed Episodes - Agent: {agent_name}, Memory: {memory_value}")
            
        metrics["resource_usage"][episodes_agent_name] = resource_data
        
    for agent_name, data in fixed_resources.items():
        resource_agent_name = f"{agent_name} (Fixed Resource)"
        metrics["agent_performance"][resource_agent_name] = {"win_rate": data.get("win_rate", 0)}
        
        agent_perf = data.get("agent_performance", {}).get(agent_name, {})
        if agent_perf:
            metrics["agent_performance"][resource_agent_name].update(agent_perf)
        
        resource_data = data.get("resource_usage", {}).get(agent_name, {})
        memory_value = resource_data.get("avg_memory_mb", 0)
        print(f"Fixed Resource - Agent: {agent_name}, Memory: {memory_value}")
            
        metrics["resource_usage"][resource_agent_name] = resource_data

        
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Generated metrics file: {metrics_path}")
    print("Resource usage data sample:")
    for agent, data in list(metrics["resource_usage"].items())[:2]:
        print(f"  {agent}: {data}")

    return metrics_path

def run_pareto_analysis(evaluation_dir, output_dir=None, n_latest=5, show_size_values=False, use_multiple_experiments=False):
    """
    Run Pareto analysis on evaluation data
    
    Args:
        evaluation_dir: Directory containing evaluation data
        output_dir: Directory to save visualizations
        n_latest: Number of latest experiments to analyze (for traditional mode)
        show_size_values: Whether to show size values on plots
        use_multiple_experiments: Whether to use multiple experiment mode
        
    Returns:
        bool: Success flag
    """
    if evaluation_dir.startswith('./') or evaluation_dir.startswith('../'):
        project_root = find_project_root()
        if evaluation_dir == './evaluations':
            evaluation_dir = os.path.join(project_root, 'evaluations')
            print(f"Using project root evaluations directory: {evaluation_dir}")
    
    if use_multiple_experiments:
        print("Using multiple experiments mode for Pareto analysis")
        df = aggregate_experiment_data(evaluation_dir, n_latest, use_multiple_experiments=True)

    else:
        report_files = glob.glob(os.path.join(evaluation_dir, "integrated_evaluation_report_*.json"))

        
        latest_report = max(report_files, key=os.path.getctime)
        print(f"Using latest report: {latest_report}")
        
        adapted_dir = os.path.join(evaluation_dir, "adapted_data")
        metrics_path = convert_integrated_report_for_pareto(latest_report, adapted_dir)
        
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        
        agent_data = []
        for agent_name, metrics in metrics_data.get("agent_performance", {}).items():
            resource_usage = metrics_data.get("resource_usage", {}).get(agent_name, {})
            
            win_rate = metrics.get("win_rate", 0) * 100
            avg_time = metrics.get("avg_time_ms", 0)
            memory_mb = resource_usage.get("avg_memory_mb", 0)
            
            print(f"Agent {agent_name}: Win rate = {win_rate}%, Memory = {memory_mb} MB")
            
            agent_data.append({
                "Agent": agent_name,
                "Win Rate (%)": win_rate,
                "Avg Time (ms)": avg_time,
                "Memory (MB)": memory_mb,
                "CPU (%)": resource_usage.get("avg_cpu_percent", 0),
                "n_samples": 1 
            })
        
        df = pd.DataFrame(agent_data)
    
    if df is not None and not df.empty:
        df.loc[df["Memory (MB)"] == 0, "Memory (MB)"] = 0.1
        
        if output_dir is None:
            output_dir = os.path.join(evaluation_dir, "analysis")

        create_2d_pareto_analysis(df, output_dir, show_size_values)
        create_3d_visualization(df, output_dir)
        create_radar_chart(df, output_dir)
        print(f"Pareto analysis completed. Visualizations saved to: {output_dir}")
        return True
    
    print("No valid data for Pareto analysis.")
    return False


def run_adapted_pareto_analysis(evaluation_dir="./evaluations", show_size_values=False):
    report_files = glob.glob(os.path.join(evaluation_dir, "integrated_evaluation_report_*.json"))
    
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
            "Win Rate (%)": metrics.get("win_rate", 0) * 100,
            "Avg Time (ms)": metrics.get("avg_time_ms", 0),
            "Memory (MB)": resource_usage.get("avg_memory_mb", 0),
            "CPU (%)": resource_usage.get("avg_cpu_percent", 0),
            "n_samples": 1 
        })
    
    df = pd.DataFrame(agent_data)
    
    output_dir = os.path.join(evaluation_dir, "pareto_plots")
    create_2d_pareto_analysis(df, output_dir, show_size_values)
    create_3d_visualization(df, output_dir)
    create_radar_chart(df, output_dir)
    return True


if __name__ == "__main__":
    import sys
    
    def find_latest_experiment():
        project_root = find_project_root()
        log_dir = os.path.join(project_root, "logs")
        
        if os.path.exists(log_dir):
            log_dirs = sorted([d for d in os.listdir(log_dir) if d.startswith("experiment_")])
            if not log_dirs:
                return None
            return os.path.join(log_dir, log_dirs[-1])
        return None
    
    import argparse
    parser = argparse.ArgumentParser(description='pareto analysis')
    parser.add_argument('--dir', type=str, default="./evaluations", help='target experiment directory')
    parser.add_argument('--n_latest', type=int, default=6, help='number of latest experiments to analyze')
    parser.add_argument('--output', type=str, default=None, help='output directory for plots')
    parser.add_argument('--show-size-values', action='store_true', help='show additional metric values on plots')
    parser.add_argument('--multiple', action='store_true', help='use multiple experiments data integration')
    
    args = parser.parse_args()
    
    project_root = find_project_root()
    
    if args.dir == "./evaluations":
        results_dir = os.path.join(project_root, "evaluations")
    else:
        results_dir = args.dir if args.dir else find_latest_experiment()
    
    if args.output is None:
        output_dir = os.path.join(results_dir, "analysis")
    else:
        output_dir = args.output
    
    
    success = run_pareto_analysis(
        results_dir, 
        output_dir, 
        args.n_latest, 
        args.show_size_values,
        args.multiple
    )
    
    if success:
        print(f"Pareto analysis completed successfully. Results saved to: {output_dir}")
    else:
        print("Pareto analysis failed.")
        sys.exit(1)