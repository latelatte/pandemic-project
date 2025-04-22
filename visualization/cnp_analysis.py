"""
 CNP (Cost-Normalized Performance) Analysis Script

This script extends the CNP analysis with more detailed statistical validation,
including confidence intervals and effect size calculations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
import scipy.stats as stats
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


def calculate_cnp(win_rate, memory_gb, time_hrs):
    """
    Calculate CNP (Cost-Normalized Performance)
    
    Args:
        win_rate: Win rate percentage (0-100)
        memory_gb: Memory usage in GB
        time_hrs: Time usage in hours
        
    Returns:
        float: CNP value
    """
    if memory_gb <= 0 or time_hrs <= 0:
        return 0
        
    return win_rate / (memory_gb * time_hrs)


def load_dual_evaluation_data(eval_dir, report_file=None):
    """
    Load data from dual evaluation framework
    
    Args:
        eval_dir: Directory containing evaluation reports
        report_file: Optional specific report file to load
        
    Returns:
        dict: Processed evaluation data
    """
    # If a specific file is provided, use it; otherwise find the latest
    if report_file:
        latest_report = report_file
        print(f"Loading specified evaluation data from: {latest_report}")
    else:
        # Find the latest evaluation report
        report_files = glob.glob(os.path.join(eval_dir, "*evaluation_report_*.json"))
        if not report_files:
            print("No evaluation reports to analyze CNP found.")
            return None
            
        latest_report = max(report_files, key=os.path.getctime)
        print(f"Loading evaluation data from: {latest_report}")
    
    try:
        with open(latest_report, 'r') as f:
            report_data = json.load(f)
            
        # Process the data
        processed_data = {
            "fixed_episodes": {},
            "fixed_resource": {},
            "comparison": {},
            "cnp_metrics": report_data.get("cnp_metrics", {}),
            "settings": report_data.get("settings", {})
        }
        
        # Extract fixed episodes results
        for agent, metrics in report_data.get("fixed_episodes_results", {}).items():
            win_rate = metrics.get("win_rate", 0) * 100
            avg_time = metrics.get("agent_performance", {}).get(agent, {}).get("avg_time_ms", 0)
            memory_mb = metrics.get("resource_usage", {}).get(agent, {}).get("avg_memory_mb", 0)
            
            processed_data["fixed_episodes"][agent] = {
                "win_rate": win_rate,
                "avg_time_ms": avg_time,
                "memory_mb": memory_mb,
                "episodes": report_data.get("settings", {}).get("fixed_episodes", 5000)
            }
            
        # Extract fixed resource results
        for agent, metrics in report_data.get("fixed_resource_results", {}).items():
            win_rate = metrics.get("win_rate", 0) * 100
            avg_time = metrics.get("agent_performance", {}).get(agent, {}).get("avg_time_ms", 0)
            memory_mb = metrics.get("resource_usage", {}).get(agent, {}).get("avg_memory_mb", 0)
            
            # Fix: Use "total_episodes" instead of "episodes_completed" to get the episode count
            episodes = metrics.get("total_episodes", 0)
            
            processed_data["fixed_resource"][agent] = {
                "win_rate": win_rate,
                "avg_time_ms": avg_time,
                "memory_mb": memory_mb,
                "episodes": episodes
            }
            
        # Extract comparison data
        processed_data["comparison"] = report_data.get("comparison_summary", {})
            
        return processed_data
        
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return None


def find_project_root():
    """
    プロジェクトルートディレクトリを自動検出する
    
    Returns:
        str: プロジェクトルートのパス
    """
    # 現在のスクリプトの絶対パス
    current_path = os.path.abspath(__file__)
    
    # 現在のスクリプトのディレクトリ
    current_dir = os.path.dirname(current_path)
    
    # 親ディレクトリ (おそらくプロジェクトルート)
    parent_dir = os.path.dirname(current_dir)
    
    # プロジェクトルートの検証 (evaluationsディレクトリが存在するか確認)
    if os.path.exists(os.path.join(parent_dir, "evaluations")):
        print(f"Found project root: {parent_dir}")
        return parent_dir
    
    # evaluationsが見つからない場合は、現在のディレクトリを返す (フォールバック)
    print(f"Project root not detected. Using current directory: {current_dir}")
    return current_dir


def load_multiple_experiment_data(eval_dir, experiment_dirs=None):
    """
    Load data from multiple independent experiments
    
    Args:
        eval_dir: Base directory containing experiment directories
        experiment_dirs: List of specific experiment directories to load
        
    Returns:
        list: List of processed evaluation data from each experiment
    """
    all_experiments_data = []
    
    # 相対パスが指定された場合はプロジェクトルートを考慮
    if eval_dir.startswith('./') or eval_dir.startswith('../'):
        project_root = find_project_root()
        # ./evaluations が指定された場合、プロジェクトルートの evaluations を使用
        if eval_dir == './evaluations':
            eval_dir = os.path.join(project_root, 'evaluations')
            print(f"Using project root evaluations directory: {eval_dir}")
    
    # ディレクトリパスを絶対パスに変換して存在確認
    eval_dir = os.path.abspath(eval_dir)
    print(f"Using base directory: {eval_dir}")
    
    if not os.path.exists(eval_dir):
        print(f"Error: Directory '{eval_dir}' does not exist.")
        # カレントディレクトリの確認
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")
        # サブディレクトリの一覧表示
        if os.path.exists(os.path.dirname(eval_dir)):
            parent_dir = os.path.dirname(eval_dir)
            print(f"Contents of parent directory ({parent_dir}):")
            for item in os.listdir(parent_dir):
                print(f"  - {item}")
        return []
    
    # リポジトリ内のすべての評価レポートを読み込む
    all_report_files = []
    
    # ディレクトリ構造を探索
    if experiment_dirs:
        # 特定のディレクトリが指定された場合
        for exp_dir_name in experiment_dirs:
            exp_dir = os.path.join(eval_dir, exp_dir_name)
            if os.path.exists(exp_dir) and os.path.isdir(exp_dir):
                report_files = glob.glob(os.path.join(exp_dir, "*evaluation_report_*.json"))
                # 統合済みファイルを除外しない（すべてのレポートを処理）
                all_report_files.extend(report_files)
    else:
        # 自動探索モード - 直接ディレクトリ内のレポートを探す
        report_files = glob.glob(os.path.join(eval_dir, "*evaluation_report_*.json"))
        if report_files:
            # レポートをタイムスタンプでソート
            report_files = sorted(report_files, key=os.path.getctime)
            all_report_files.extend(report_files)
            print(f"Found {len(report_files)} evaluation reports directly in {eval_dir}")
        
        # サブディレクトリも探索
        for root, dirs, files in os.walk(eval_dir):
            # 現在のディレクトリがeval_dirそのものでない場合のみ処理
            if root != eval_dir:
                for file in files:
                    if file.endswith(".json") and "evaluation_report" in file:
                        all_report_files.append(os.path.join(root, file))
    
    # 重複を削除
    all_report_files = list(set(all_report_files))
    
    print(f"Found {len(all_report_files)} evaluation reports:")
    for report_file in all_report_files[:5]:  # 最初の5つだけ表示
        print(f"  - {os.path.basename(report_file)}")
    if len(all_report_files) > 5:
        print(f"  ... and {len(all_report_files) - 5} more")
    
    # ここが重要なポイント: 各レポートを個別に処理
    for report_file in all_report_files:
        try:
            print(f"Processing report: {report_file}")
            exp_data = load_dual_evaluation_data(os.path.dirname(report_file), report_file)
            
            if exp_data:
                all_experiments_data.append(exp_data)
                print(f"Added data from report: {os.path.basename(report_file)}")
                
                # エージェント情報を出力して確認
                for agent_type in ["fixed_episodes", "fixed_resource"]:
                    if agent_type in exp_data:
                        print(f"  {agent_type.upper()} agents:")
                        for agent, data in exp_data[agent_type].items():
                            win_rate = data.get("win_rate", 0)
                            print(f"    - {agent}: Win rate = {win_rate:.2f}%")
            else:
                print(f"Failed to load data from {report_file}")
        except Exception as e:
            print(f"Error processing file {report_file}: {e}")
    
    print(f"Successfully loaded data from {len(all_experiments_data)} experiment reports")
    return all_experiments_data


def integrate_experiment_data(all_experiments_data):
    """
    Integrate data from multiple experiments
    
    Args:
        all_experiments_data: List of data from individual experiments
        
    Returns:
        dict: Integrated data with means and statistical measures
    """
    if not all_experiments_data:
        print("No experiment data to integrate!")
        return None
    
    # Initialize the integrated data structure
    integrated_data = {
        "fixed_episodes": {},
        "fixed_resource": {},
        "comparison": {},
        "cnp_metrics": {},
        "statistics": {
            "fixed_episodes": {},
            "fixed_resource": {},
            "cnp_metrics": {}
        },
        "settings": all_experiments_data[0].get("settings", {})
    }
    
    # Get all unique agents across experiments
    all_agents = set()
    for exp_data in all_experiments_data:
        all_agents.update(exp_data.get("fixed_episodes", {}).keys())
        all_agents.update(exp_data.get("fixed_resource", {}).keys())
    
    # Process fixed episodes data
    for agent in all_agents:
        win_rates = []
        avg_times = []
        memory_mbs = []
        episodes = []
        cnp_values = []
        
        for exp_data in all_experiments_data:
            if agent in exp_data.get("fixed_episodes", {}):
                agent_data = exp_data["fixed_episodes"][agent]
                win_rates.append(agent_data.get("win_rate", 0))
                avg_times.append(agent_data.get("avg_time_ms", 0))
                memory_mbs.append(agent_data.get("memory_mb", 0))
                episodes.append(agent_data.get("episodes", 0))
                
                # Get CNP values if available
                if "cnp_metrics" in exp_data and agent in exp_data["cnp_metrics"]:
                    cnp_values.append(exp_data["cnp_metrics"][agent].get("fixed_episodes_cnp", 0))
        
        # 各エクスペリメントのデータをデバッグ表示
        print(f"Agent {agent} data from all experiments:")
        for i, exp_data in enumerate(all_experiments_data):
            if agent in exp_data.get("fixed_episodes", {}):
                agent_data = exp_data["fixed_episodes"][agent]
                win_rate = agent_data.get("win_rate", 0)
                print(f"  Experiment {i+1}: Fixed Episodes Win Rate = {win_rate:.2f}%")
        
        # Calculate statistics if we have data
        if win_rates:
            # Mean values
            mean_win_rate = np.mean(win_rates)
            mean_avg_time = np.mean(avg_times)
            mean_memory_mb = np.mean(memory_mbs)
            mean_episodes = np.mean(episodes)
            
            # Standard deviations
            std_win_rate = np.std(win_rates)
            std_avg_time = np.std(avg_times)
            std_memory_mb = np.std(memory_mbs)
            
            # 95% confidence intervals
            if len(win_rates) > 1:
                ci_win_rate = stats.t.interval(
                    0.95, len(win_rates)-1, 
                    loc=mean_win_rate, 
                    scale=stats.sem(win_rates)
                )
            else:
                # If only one data point, set CI to the same value
                ci_win_rate = (mean_win_rate, mean_win_rate)
            
            # Store mean values in the main structure
            integrated_data["fixed_episodes"][agent] = {
                "win_rate": mean_win_rate,
                "avg_time_ms": mean_avg_time,
                "memory_mb": mean_memory_mb,
                "episodes": mean_episodes
            }
            
            # Store statistical data
            integrated_data["statistics"]["fixed_episodes"][agent] = {
                "win_rate": {
                    "values": win_rates,
                    "mean": mean_win_rate,
                    "std": std_win_rate,
                    "ci_low": ci_win_rate[0],
                    "ci_high": ci_win_rate[1]
                },
                "avg_time_ms": {
                    "values": avg_times,
                    "mean": mean_avg_time,
                    "std": std_avg_time
                },
                "memory_mb": {
                    "values": memory_mbs,
                    "mean": mean_memory_mb,
                    "std": std_memory_mb
                }
            }
            
            # Store CNP statistics if available
            if cnp_values:
                mean_cnp = np.mean(cnp_values)
                std_cnp = np.std(cnp_values) if len(cnp_values) > 1 else 0
                
                if agent not in integrated_data["cnp_metrics"]:
                    integrated_data["cnp_metrics"][agent] = {}
                    
                integrated_data["cnp_metrics"][agent]["fixed_episodes_cnp"] = mean_cnp
                
                integrated_data["statistics"]["cnp_metrics"][agent] = {
                    "fixed_episodes_cnp": {
                        "values": cnp_values,
                        "mean": mean_cnp,
                        "std": std_cnp
                    }
                }
            
            # デバッグ出力を追加
            print(f"Agent {agent} integrated statistics:")
            print(f"  Fixed Episodes: {len(win_rates)} values, mean={mean_win_rate:.2f}%, " +
                  f"CI=[{ci_win_rate[0]:.2f}, {ci_win_rate[1]:.2f}]")
            print(f"    Raw values: {win_rates}")
    
    # Process fixed resource data
    for agent in all_agents:
        win_rates = []
        avg_times = []
        memory_mbs = []
        episodes = []
        cnp_values = []
        
        for exp_data in all_experiments_data:
            if agent in exp_data.get("fixed_resource", {}):
                agent_data = exp_data["fixed_resource"][agent]
                win_rates.append(agent_data.get("win_rate", 0))
                avg_times.append(agent_data.get("avg_time_ms", 0))
                memory_mbs.append(agent_data.get("memory_mb", 0))
                episodes.append(agent_data.get("episodes", 0))
                
                # Get CNP values if available
                if "cnp_metrics" in exp_data and agent in exp_data["cnp_metrics"]:
                    cnp_values.append(exp_data["cnp_metrics"][agent].get("fixed_resource_cnp", 0))
        
        # 各エクスペリメントのデータをデバッグ表示
        for i, exp_data in enumerate(all_experiments_data):
            if agent in exp_data.get("fixed_resource", {}):
                agent_data = exp_data["fixed_resource"][agent]
                win_rate = agent_data.get("win_rate", 0)
                print(f"  Experiment {i+1}: Fixed Resource Win Rate = {win_rate:.2f}%")
        
        # Calculate statistics if we have data
        if win_rates:
            # Mean values
            mean_win_rate = np.mean(win_rates)
            mean_avg_time = np.mean(avg_times)
            mean_memory_mb = np.mean(memory_mbs)
            mean_episodes = np.mean(episodes)
            
            # Standard deviations
            std_win_rate = np.std(win_rates)
            std_avg_time = np.std(avg_times)
            std_memory_mb = np.std(memory_mbs)
            std_episodes = np.std(episodes)
            
            # 95% confidence intervals
            if len(win_rates) > 1:
                ci_win_rate = stats.t.interval(
                    0.95, len(win_rates)-1, 
                    loc=mean_win_rate, 
                    scale=stats.sem(win_rates)
                )
                ci_episodes = stats.t.interval(
                    0.95, len(episodes)-1, 
                    loc=mean_episodes, 
                    scale=stats.sem(episodes)
                )
            else:
                # If only one data point, set CI to the same value
                ci_win_rate = (mean_win_rate, mean_win_rate)
                ci_episodes = (mean_episodes, mean_episodes)
            
            # Store mean values in the main structure
            integrated_data["fixed_resource"][agent] = {
                "win_rate": mean_win_rate,
                "avg_time_ms": mean_avg_time,
                "memory_mb": mean_memory_mb,
                "episodes": mean_episodes
            }
            
            # Store statistical data
            integrated_data["statistics"]["fixed_resource"][agent] = {
                "win_rate": {
                    "values": win_rates,
                    "mean": mean_win_rate,
                    "std": std_win_rate,
                    "ci_low": ci_win_rate[0],
                    "ci_high": ci_win_rate[1]
                },
                "avg_time_ms": {
                    "values": avg_times,
                    "mean": mean_avg_time,
                    "std": std_avg_time
                },
                "memory_mb": {
                    "values": memory_mbs,
                    "mean": mean_memory_mb,
                    "std": std_memory_mb
                },
                "episodes": {
                    "values": episodes,
                    "mean": mean_episodes,
                    "std": std_episodes,
                    "ci_low": ci_episodes[0],
                    "ci_high": ci_episodes[1]
                }
            }
            
            # Store CNP statistics if available
            if cnp_values:
                mean_cnp = np.mean(cnp_values)
                std_cnp = np.std(cnp_values) if len(cnp_values) > 1 else 0
                
                if agent not in integrated_data["cnp_metrics"]:
                    integrated_data["cnp_metrics"][agent] = {}
                    
                integrated_data["cnp_metrics"][agent]["fixed_resource_cnp"] = mean_cnp
                
                if agent not in integrated_data["statistics"]["cnp_metrics"]:
                    integrated_data["statistics"]["cnp_metrics"][agent] = {}
                
                integrated_data["statistics"]["cnp_metrics"][agent]["fixed_resource_cnp"] = {
                    "values": cnp_values,
                    "mean": mean_cnp,
                    "std": std_cnp
                }
    
    return integrated_data


def create_cnp_visualization(data, output_dir):
    """
    Create CNP visualizations
    
    Args:
        data: Processed evaluation data
        output_dir: Directory to save visualizations
        
    Returns:
        bool: Success flag
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for visualization
    df_episodes = []
    for agent, metrics in data["fixed_episodes"].items():
        df_episodes.append({
            "Agent": agent,
            "Evaluation": "Fixed Episodes",
            "Win Rate (%)": metrics["win_rate"],
            "Avg Time (ms)": metrics["avg_time_ms"],
            "Memory (MB)": metrics["memory_mb"],
            "Episodes": metrics["episodes"],
            "CNP": data["cnp_metrics"].get(agent, {}).get("fixed_episodes_cnp", 0)
        })
        
    df_resource = []
    for agent, metrics in data["fixed_resource"].items():
        df_resource.append({
            "Agent": agent,
            "Evaluation": "Fixed Resource",
            "Win Rate (%)": metrics["win_rate"],
            "Avg Time (ms)": metrics["avg_time_ms"],
            "Memory (MB)": metrics["memory_mb"],
            "Episodes": metrics["episodes"],
            "CNP": data["cnp_metrics"].get(agent, {}).get("fixed_resource_cnp", 0)
        })
    
    df = pd.DataFrame(df_episodes + df_resource)
    
    # 1. Create dual evaluation comparison chart
    plt.figure(figsize=(12, 8))
    
    # Define colors for each agent and evaluation type
    unique_agents = df["Agent"].unique()
    agent_colors = sns.color_palette("husl", len(unique_agents))
    color_map = {agent: color for agent, color in zip(unique_agents, agent_colors)}
    
    marker_styles = {"Fixed Episodes": "o", "Fixed Resource": "s"}
    
    # Plot win rate vs. resource usage
    for i, agent in enumerate(unique_agents):
        for eval_type in ["Fixed Episodes", "Fixed Resource"]:
            agent_data = df[(df["Agent"] == agent) & (df["Evaluation"] == eval_type)]
            
            if not agent_data.empty:
                plt.scatter(
                    agent_data["Avg Time (ms)"], 
                    agent_data["Memory (MB)"],
                    s=agent_data["Win Rate (%)"] * 5,  # Size proportional to win rate
                    color=color_map[agent],
                    marker=marker_styles[eval_type],
                    alpha=0.7,
                    label=f"{agent} ({eval_type})"
                )
                
                # Annotate with win rate
                for _, row in agent_data.iterrows():
                    plt.annotate(
                        f"{row['Win Rate (%)']:.1f}%", 
                        (row["Avg Time (ms)"], row["Memory (MB)"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9
                    )
                    
                    # Add episode count annotation with error checking
                    episodes_text = f"{int(row['Episodes'])} ep" if not pd.isna(row['Episodes']) and row['Episodes'] > 0 else "N/A ep"
                    plt.annotate(
                        episodes_text, 
                        (row["Avg Time (ms)"], row["Memory (MB)"]),
                        xytext=(5, -10),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7
                    )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Average Decision Time (ms) - log scale", fontsize=12)
    plt.ylabel("Memory Usage (MB) - log scale", fontsize=12)
    plt.title("Dual Evaluation: Resource Usage vs. Performance", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Add contour lines for CNP values
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    cnp_contours = [0.1, 1, 10, 100, 1000]
    x = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    
    for cnp in cnp_contours:
        reference_win_rate = 50
        y = reference_win_rate / (cnp * (x / 3600000)) * 1024  # Convert time to hours and memory to GB
        valid_indices = (y >= y_min) & (y <= y_max)
        if np.any(valid_indices):
            plt.plot(x[valid_indices], y[valid_indices], '--', color='gray', alpha=0.5)
            
            # Label the contour line
            mid_idx = np.where(valid_indices)[0][len(np.where(valid_indices)[0])//2]
            plt.text(x[mid_idx], y[mid_idx], f"CNP={cnp}", 
                    fontsize=8, color='gray', alpha=0.8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dual_evaluation_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Create CNP comparison chart - Both linear and log scale versions
    unique_agents = df["Agent"].unique()
    
    # 2.1 Linear scale version
    plt.figure(figsize=(10, 6))
    cnp_data = []
    for agent in unique_agents:
        fixed_ep_cnp = data["cnp_metrics"].get(agent, {}).get("fixed_episodes_cnp", 0)
        fixed_res_cnp = data["cnp_metrics"].get(agent, {}).get("fixed_resource_cnp", 0)
        cnp_data.append({
            "Agent": agent,
            "Fixed Episodes CNP": fixed_ep_cnp,
            "Fixed Resource CNP": fixed_res_cnp,
            "CNP Ratio": fixed_res_cnp / max(0.001, fixed_ep_cnp)
        })
    cnp_df = pd.DataFrame(cnp_data)
    
    bar_width = 0.35
    x = np.arange(len(unique_agents))
    plt.bar(x - bar_width/2, cnp_df["Fixed Episodes CNP"], 
            width=bar_width, color='skyblue', label='Fixed Episodes')
    plt.bar(x + bar_width/2, cnp_df["Fixed Resource CNP"], 
            width=bar_width, color='salmon', label='Fixed Resource')
        
    for i, value in enumerate(cnp_df["Fixed Episodes CNP"]):
        plt.text(i - bar_width/2, value + 0.1, f"{value:.1f}", 
                ha='center', va='bottom', fontsize=9)
    for i, value in enumerate(cnp_df["Fixed Resource CNP"]):
        plt.text(i + bar_width/2, value + 0.1, f"{value:.1f}", 
                ha='center', va='bottom', fontsize=9)
    
    plt.xlabel("Agent", fontsize=12)
    plt.ylabel("CNP Value", fontsize=12)
    plt.title("Cost-Normalized Performance Comparison (Linear Scale)", fontsize=16)
    plt.xticks(x, unique_agents)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for i, ratio in enumerate(cnp_df["CNP Ratio"]):
        plt.annotate(
            f"Ratio: {ratio:.2f}x", 
            (x[i], 0.1),
            xytext=(0, -30),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc='lightyellow', alpha=0.7)
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cnp_comparison_linear.png"), dpi=300)
    plt.close()
    
    # 2.2 Log scale version
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - bar_width/2, cnp_df["Fixed Episodes CNP"], 
            width=bar_width, color='skyblue', label='Fixed Episodes')
    bars2 = plt.bar(x + bar_width/2, cnp_df["Fixed Resource CNP"], 
            width=bar_width, color='salmon', label='Fixed Resource')
    for i, value in enumerate(cnp_df["Fixed Episodes CNP"]):
        y_pos = max(value * 1.1, 0.1)
        plt.text(i - bar_width/2, y_pos, f"{value:.1f}", 
                ha='center', va='bottom', fontsize=9, rotation=0,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
    for i, value in enumerate(cnp_df["Fixed Resource CNP"]):
        y_pos = max(value * 1.1, 0.1)
        plt.text(i + bar_width/2, y_pos, f"{value:.1f}", 
                ha='center', va='bottom', fontsize=9, rotation=0,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.xlabel("Agent", fontsize=12)
    plt.ylabel("CNP Value (log scale)", fontsize=12)
    plt.title("Cost-Normalized Performance Comparison (Log Scale)", fontsize=16)
    plt.yscale('log')
    plt.xticks(x, unique_agents)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for i, ratio in enumerate(cnp_df["CNP Ratio"]):
        y_min, y_max = plt.ylim()
        y_pos = y_min * (y_max/y_min)**0.1
        plt.annotate(
            f"Ratio: {ratio:.2f}x", 
            (x[i], y_pos),
            xytext=(0, 10),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc='lightyellow', alpha=0.7)
        )
    plt.figtext(0.01, 0.01, "Note: Y-axis uses logarithmic scale for better visibility", 
                fontsize=8, style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cnp_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Create win rate vs. episode count chart for fixed resource evaluation
    resource_df = pd.DataFrame(df_resource)
    
    # 図のサイズを少し大きくして、タイトルが見切れないようにする
    plt.figure(figsize=(12, 8))
    
    scatter = sns.scatterplot(
        data=resource_df,
        x="Episodes",
        y="Win Rate (%)",
        hue="Agent",
        s=200,
        alpha=0.7
    )
    
    if scatter.get_legend():
        scatter.get_legend().remove()
    
    plt.legend(title="Agent", loc="best", framealpha=0.9)

    for i, row in resource_df.iterrows():
        plt.annotate(
            f"CNP: {row['CNP']:.3f}", 
            (row["Episodes"], row["Win Rate (%)"]),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
        )
    
    plt.xlabel("Episodes Completed Under Resource Constraint", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.title("Fixed Resource Evaluation: Win Rate vs. Episode Count", fontsize=16, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "fixed_resource_episodes.png"), dpi=300)
    plt.close()
    
    # 4. Create breakdown comparison view
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax_main = plt.subplot(gs[0])
    bars1 = ax_main.bar(x - bar_width/2, cnp_df["Fixed Episodes CNP"], 
            width=bar_width, color='skyblue', label='Fixed Episodes')
    bars2 = ax_main.bar(x + bar_width/2, cnp_df["Fixed Resource CNP"], 
            width=bar_width, color='salmon', label='Fixed Resource')
    for i, value in enumerate(cnp_df["Fixed Episodes CNP"]):
        y_pos = max(value * 1.1, 0.1)
        ax_main.text(i - bar_width/2, y_pos, f"{value:.1f}", 
                ha='center', va='bottom', fontsize=9)
    for i, value in enumerate(cnp_df["Fixed Resource CNP"]):
        y_pos = max(value * 1.1, 0.1)
        ax_main.text(i + bar_width/2, y_pos, f"{value:.1f}", 
                ha='center', va='bottom', fontsize=9)
    ax_main.set_xlabel("Agent", fontsize=12)
    ax_main.set_ylabel("CNP Value (log scale)", fontsize=12)
    ax_main.set_title("Full Comparison (Log Scale)", fontsize=14)
    ax_main.set_yscale('log')
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(unique_agents)
    ax_main.legend()
    ax_main.grid(True, axis='y', linestyle='--', alpha=0.7)
    max_cnp_idx = cnp_df["Fixed Episodes CNP"].idxmax()
    max_agent = cnp_df.iloc[max_cnp_idx]["Agent"]
    ax_detail = plt.subplot(gs[1])
    detail_df = cnp_df[cnp_df["Agent"] != max_agent].reset_index(drop=True)
    detail_x = np.arange(len(detail_df))
    bars3 = ax_detail.bar(detail_x - bar_width/2, detail_df["Fixed Episodes CNP"], 
            width=bar_width, color='skyblue')
    bars4 = ax_detail.bar(detail_x + bar_width/2, detail_df["Fixed Resource CNP"], 
            width=bar_width, color='salmon')
    for i, value in enumerate(detail_df["Fixed Episodes CNP"]):
        ax_detail.text(i - bar_width/2, value * 1.05, f"{value:.2f}", 
                ha='center', va='bottom', fontsize=8)
    for i, value in enumerate(detail_df["Fixed Resource CNP"]):
        ax_detail.text(i + bar_width/2, value * 1.05, f"{value:.2f}", 
                ha='center', va='bottom', fontsize=8)
    ax_detail.set_xlabel("Agent", fontsize=10)
    ax_detail.set_title(f"Detail View (Without {max_agent})", fontsize=14)
    ax_detail.set_xticks(detail_x)
    ax_detail.set_xticklabels(detail_df["Agent"])
    ax_detail.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cnp_comparison_breakdown.png"), dpi=300)
    plt.close()
    
    # 5. Create efficiency comparison chart
    plt.figure(figsize=(12, 8))
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
    
    ax_main = plt.subplot(gs[1, 0])
    
    scatter = ax_main.scatter(
        df["Avg Time (ms)"],
        df["Memory (MB)"],
        c=df["Win Rate (%)"],
        s=100,
        cmap="viridis",
        alpha=0.8
    )
    
    for i, row in df.iterrows():
        ax_main.annotate(
            f"{row['Agent']} ({row['Evaluation']})", 
            (row["Avg Time (ms)"], row["Memory (MB)"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )
    
    ax_main.set_xlabel("Average Decision Time (ms)", fontsize=12)
    ax_main.set_ylabel("Memory Usage (MB)", fontsize=12)
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.grid(True, linestyle='--', alpha=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label("Win Rate (%)")
    
    ax_time = plt.subplot(gs[0, 0], sharex=ax_main)
    
    df["Time Efficiency"] = df["Win Rate (%)"] / df["Avg Time (ms)"]
    
    # Time Efficiencyグラフを対数スケールに変更し、全エージェントを表示
    ax_time.set_yscale('log')  # 対数スケールを適用
    
    # エージェントごとにグループ化して横並びに表示するよう調整
    n_agents = len(unique_agents)
    
    # エージェントの表示位置を決定するオフセット
    time_offsets = {}
    
    for i, agent in enumerate(unique_agents):
        agent_data = df[df["Agent"] == agent]
        
        # 各エージェントデータの平均時間を計算して基準位置を求める
        if not agent_data.empty:
            time_val = agent_data["Avg Time (ms)"].mean()
            time_offsets[agent] = time_val
    
    # 全バーで使用する固定幅の計算（対数スケール上で適切に見えるサイズ）
    fixed_width = 0.3  # 固定幅
    
    # 固定エピソードと固定リソースを横並びに表示
    for i, agent in enumerate(unique_agents):
        agent_data = df[df["Agent"] == agent]
        
        for j, eval_type in enumerate(["Fixed Episodes", "Fixed Resource"]):
            eval_data = agent_data[agent_data["Evaluation"] == eval_type]
            
            if not eval_data.empty:
                # 基準位置からオフセットを計算
                time_val = time_offsets[agent]
                # 評価タイプによって左右にずらす
                offset = 0.7 if j == 0 else 1.3
                
                # 対数スケールなので値が0の場合に備える
                efficiency = max(eval_data["Time Efficiency"].iloc[0], 1e-10)
                
                bar = ax_time.bar(
                    time_val * offset,
                    efficiency,
                    width=fixed_width * time_val,  # 幅は固定係数と時間の積
                    color=color_map[agent],
                    alpha=0.7 if eval_type == "Fixed Episodes" else 0.4,
                    label=f"{agent} ({eval_type})" if i == 0 else None
                )
                
    ax_time.set_ylabel("Time Efficiency\n(Win Rate / Time) - log scale", fontsize=10)
    ax_time.set_xscale('log')
    ax_time.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax_time.tick_params(axis='x', labelsize=0)
    
    ax_memory = plt.subplot(gs[1, 1], sharey=ax_main)
    
    df["Memory Efficiency"] = df["Win Rate (%)"] / df["Memory (MB)"]
    
    # Memory Efficiencyグラフを横並びに表示
    # エージェントの表示位置を決定するオフセット
    memory_offsets = {}
    
    for i, agent in enumerate(unique_agents):
        agent_data = df[df["Agent"] == agent]
        
        # 各エージェントデータの平均メモリを計算して基準位置を求める
        if not agent_data.empty:
            memory_val = agent_data["Memory (MB)"].mean()
            memory_offsets[agent] = memory_val
    
    # 固定エピソードと固定リソースを横並びに表示
    for i, agent in enumerate(unique_agents):
        agent_data = df[df["Agent"] == agent]
        
        for j, eval_type in enumerate(["Fixed Episodes", "Fixed Resource"]):
            eval_data = agent_data[agent_data["Evaluation"] == eval_type]
            
            if not eval_data.empty:
                # 基準位置からオフセットを計算
                memory_val = memory_offsets[agent]
                # 評価タイプによって左右にずらす
                offset = 0.7 if j == 0 else 1.3
                
                # 高さを固定値に設定
                fixed_height = memory_val * 0.2
                
                ax_memory.barh(
                    memory_val * offset,  # オフセットを適用して横並び
                    eval_data["Memory Efficiency"].iloc[0],
                    height=fixed_height,
                    color=color_map[agent],
                    alpha=0.7 if eval_type == "Fixed Episodes" else 0.4
                )
    
    ax_memory.set_xlabel("Memory Efficiency\n(Win Rate / Memory)", fontsize=10)
    ax_memory.set_yscale('log')
    ax_memory.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax_memory.tick_params(axis='y', labelsize=0)
    
    ax_legend = plt.subplot(gs[0, 1])
    ax_legend.axis('off')

    legend_y_start = 0.95
    legend_y_step = 0.13 
    legend_x = 0.1
    

    ax_legend.text(0.5, legend_y_start, "Agents:", fontsize=10, fontweight='bold', ha='center')
    
    for i, agent in enumerate(unique_agents):
        y_pos = legend_y_start - (i+1)*legend_y_step
        ax_legend.add_patch(Rectangle((legend_x, y_pos), 0.2, 0.1, color=color_map[agent]))
        ax_legend.text(legend_x + 0.25, y_pos + 0.05, agent, fontsize=9, va='center')
    
    eval_y_start = legend_y_start - (len(unique_agents)+1)*legend_y_step
    ax_legend.text(0.5, eval_y_start, "Evaluation Types:", fontsize=10, fontweight='bold', ha='center')
    
    # 評価方法の透明度の違いを明示する凡例
    fe_y = eval_y_start - legend_y_step
    ax_legend.add_patch(Rectangle((legend_x, fe_y), 0.2, 0.1, color=color_map[agent], alpha=0.7))
    ax_legend.text(legend_x + 0.25, fe_y + 0.05, "Fixed Episodes", fontsize=9, va='center')
    
    fr_y = eval_y_start - 2*legend_y_step
    ax_legend.add_patch(Rectangle((legend_x, fr_y), 0.2, 0.1, color=color_map[agent], alpha=0.4))
    ax_legend.text(legend_x + 0.25, fr_y + 0.05, "Fixed Resource", fontsize=9, va='center')
    

    
    plt.suptitle("Multi-dimensional Efficiency Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "efficiency_analysis.png"), dpi=300)
    plt.close()
    
    # 5. Create statistical comparison chart
    plt.figure(figsize=(12, 8))  # 元のサイズに戻す (14, 10)から(12, 8)へ
    
    stats_data = []
    
    for agent in unique_agents:
        # Use actual experimental data instead of random data
        if "statistics" in data:
            # Get data from multiple experiments if available
            ep_win_rate = data["fixed_episodes"].get(agent, {}).get("win_rate", 50)
            res_win_rate = data["fixed_resource"].get(agent, {}).get("win_rate", 50)
            
            # Get real statistical values
            ep_stats = data["statistics"]["fixed_episodes"].get(agent, {}).get("win_rate", {})
            res_stats = data["statistics"]["fixed_resource"].get(agent, {}).get("win_rate", {})
            
            ep_ci_low = ep_stats.get("ci_low", ep_win_rate)
            ep_ci_high = ep_stats.get("ci_high", ep_win_rate)
            res_ci_low = res_stats.get("ci_low", res_win_rate)
            res_ci_high = res_stats.get("ci_high", res_win_rate)

            # Get actual win rate values from experiments for t-test
            ep_wins = ep_stats.get("values", [ep_win_rate])
            res_wins = res_stats.get("values", [res_win_rate])
            
            print(f"Agent {agent} statistics for visualization:")
            print(f"  Fixed Episodes: Win rate = {ep_win_rate:.2f}%, CI = [{ep_ci_low:.2f}, {ep_ci_high:.2f}]")
            print(f"    Raw values: {ep_wins}")
            print(f"  Fixed Resource: Win rate = {res_win_rate:.2f}%, CI = [{res_ci_low:.2f}, {res_ci_high:.2f}]")
            print(f"    Raw values: {res_wins}")
        else:
            # Fallback to the original implementation with random data
            np.random.seed(42)
            ep_win_rate = data["fixed_episodes"].get(agent, {}).get("win_rate", 50)
            res_win_rate = data["fixed_resource"].get(agent, {}).get("win_rate", 50)
            
            ep_wins = np.random.normal(ep_win_rate, ep_win_rate * 0.05, 30)
            res_wins = np.random.normal(res_win_rate, res_win_rate * 0.05, 30)
            
            # Calculate confidence intervals from random data
            ep_ci = stats.t.interval(0.95, len(ep_wins)-1, loc=np.mean(ep_wins), scale=stats.sem(ep_wins))
            res_ci = stats.t.interval(0.95, len(res_wins)-1, loc=np.mean(res_wins), scale=stats.sem(res_wins))
            
            ep_ci_low = ep_ci[0]
            ep_ci_high = ep_ci[1]
            res_ci_low = res_ci[0]
            res_ci_high = res_ci[1]
            
            print(f"Warning: Using simulated data for statistical analysis of agent {agent}")
        
        # Calculate statistical significance with available data
        if len(ep_wins) > 1 and len(res_wins) > 1:
            t_stat, p_value = stats.ttest_ind(ep_wins, res_wins)
            cohens_d = (np.mean(res_wins) - np.mean(ep_wins)) / np.sqrt((np.std(ep_wins)**2 + np.std(res_wins)**2) / 2)
        else:
            # Cannot perform t-test with only one sample
            p_value = 1.0
            cohens_d = 0.0
        
        stats_data.append({
            "Agent": agent,
            "Fixed Episodes Win": ep_win_rate,
            "Fixed Episodes CI Low": ep_ci_low,
            "Fixed Episodes CI High": ep_ci_high,
            "Fixed Resource Win": res_win_rate,
            "Fixed Resource CI Low": res_ci_low,
            "Fixed Resource CI High": res_ci_high,
            "p_value": p_value,
            "cohens_d": cohens_d
        })
    
    # statsデータフレームを作成
    stats_df = pd.DataFrame(stats_data)
    
    x = np.arange(len(unique_agents))
    width = 0.35
    
    # エラーバーの計算を修正
    ep_yerr = [
        [max(0, row["Fixed Episodes Win"] - max(0, row["Fixed Episodes CI Low"])) for _, row in stats_df.iterrows()],
        [max(0, row["Fixed Episodes CI High"] - row["Fixed Episodes Win"]) for _, row in stats_df.iterrows()]
    ]
    res_yerr = [
        [max(0, row["Fixed Resource Win"] - max(0, row["Fixed Resource CI Low"])) for _, row in stats_df.iterrows()],
        [max(0, row["Fixed Resource CI High"] - row["Fixed Resource Win"]) for _, row in stats_df.iterrows()]
    ]
    
    print("Error bar values for Fixed Episodes:")
    print(f"  Lower: {ep_yerr[0]}")
    print(f"  Upper: {ep_yerr[1]}")
    
    print("Error bar values for Fixed Resource:")
    print(f"  Lower: {res_yerr[0]}")
    print(f"  Upper: {res_yerr[1]}")
    
    plt.bar(x - width/2, stats_df["Fixed Episodes Win"], width, 
            yerr=ep_yerr,
            color='skyblue', label='Fixed Episodes', capsize=5)
            
    plt.bar(x + width/2, stats_df["Fixed Resource Win"], width,
            yerr=res_yerr,
            color='salmon', label='Fixed Resource', capsize=5)
    
    # 各バーの上に勝率のラベルを追加
    for i, row in stats_df.iterrows():
        plt.text(x[i] - width/2, row["Fixed Episodes Win"] + ep_yerr[1][i] + 0.1, 
                f"{row['Fixed Episodes Win']:.2f}%", ha='center', va='bottom', fontsize=9)
        plt.text(x[i] + width/2, row["Fixed Resource Win"] + res_yerr[1][i] + 0.1, 
                f"{row['Fixed Resource Win']:.2f}%", ha='center', va='bottom', fontsize=9)
    
    for i, row in stats_df.iterrows():
        significance = ""
        if row["p_value"] < 0.001:
            significance = "***"
        elif row["p_value"] < 0.01:
            significance = "**"
        elif row["p_value"] < 0.05:
            significance = "*"
            
        if significance:
            max_y = max(row["Fixed Episodes CI High"], row["Fixed Resource CI High"])
            plt.text(x[i], max_y + 2, significance, ha='center', fontsize=14)
            
        effect_size = row["cohens_d"]
        effect_text = ""
        if abs(effect_size) < 0.2:
            effect_text = "Negligible"
        elif abs(effect_size) < 0.5:
            effect_text = "Small"
        elif abs(effect_size) < 0.8:
            effect_text = "Medium"
        else:
            effect_text = "Large"
            
        plt.text(x[i], 5, f"d={effect_size:.2f}\n({effect_text})", 
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc='lightyellow', alpha=0.7))
    
    plt.xlabel("Agent", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.title("Statistical Comparison with 95% Confidence Intervals", fontsize=16)
    plt.xticks(x, unique_agents)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # エラーバーに関する説明を追加
    if "statistics" in data:
        plt.figtext(0.01, 0.01, 
                "Error bars represent 95% confidence intervals based on multiple experiments", 
                fontsize=10, style='italic')
    
    # レイアウト調整を簡素化
    plt.tight_layout()  # 複雑な余白調整をシンプルに
    plt.savefig(os.path.join(output_dir, "statistical_comparison.png"), dpi=300)
    plt.close()
    
    return True


def run_analysis(eval_dir, output_dir=None, use_multiple_experiments=False, experiment_dirs=None):
    """
    Run the CNP analysis
    
    Args:
        eval_dir: Directory containing evaluation reports
        output_dir: Directory to save visualizations (None for auto-generation)
        use_multiple_experiments: Flag to use multiple experiment data
        experiment_dirs: List of specific experiment directories to use
        
    Returns:
        bool: Success flag
    """
    # 相対パスが指定された場合はプロジェクトルートを考慮
    if eval_dir.startswith('./') or eval_dir.startswith('../'):
        project_root = find_project_root()
        # ./evaluations が指定された場合、プロジェクトルートの evaluations を使用
        if eval_dir == './evaluations':
            eval_dir = os.path.join(project_root, 'evaluations')
            print(f"Using project root evaluations directory: {eval_dir}")

    if output_dir is None:
        output_dir = os.path.join(eval_dir, "analysis")
    
    if use_multiple_experiments:
        # Load and integrate data from multiple experiments
        print("Loading data from multiple experiments...")
        all_experiments_data = load_multiple_experiment_data(eval_dir, experiment_dirs)
        
        if not all_experiments_data:
            print("Failed to load multiple experiment data.")
            # 読み込みに失敗した場合は単一実験モードにフォールバック
            print("Falling back to single experiment mode...")
            use_multiple_experiments = False
            data = load_dual_evaluation_data(eval_dir)
            if not data:
                print("Failed to load evaluation data in fallback mode.")
                return False
        else:
            print(f"Successfully loaded {len(all_experiments_data)} experiments. Integrating data...")
            # データ統合前の詳細情報を表示
            for i, exp_data in enumerate(all_experiments_data):
                print(f"Experiment {i+1} agents:")
                for agent_type in ["fixed_episodes", "fixed_resource"]:
                    if agent_type in exp_data:
                        print(f"  {agent_type.upper()}:")
                        for agent, data in exp_data[agent_type].items():
                            win_rate = data.get("win_rate", 0)
                            print(f"    - {agent}: Win rate = {win_rate:.2f}%")
                            
            data = integrate_experiment_data(all_experiments_data)
            
            if not data:
                print("Failed to integrate experiment data.")
                return False
                
            print("Data integration complete. Creating visualizations...")
    else:
        # Use the original single experiment method
        data = load_dual_evaluation_data(eval_dir)
        if not data:
            print("Failed to load evaluation data.")
            return False
    
    success = create_cnp_visualization(data, output_dir)
    
    if success:
        if use_multiple_experiments:
            print(f"CNP analysis of multiple experiments completed. Visualizations saved to: {output_dir}")
        else:
            print(f"CNP analysis completed. Visualizations saved to: {output_dir}")
    
    return success


if __name__ == "__main__":
    import sys
    import argparse
    
    # プロジェクトルートの検出
    project_root = find_project_root()
    default_eval_dir = os.path.join(project_root, 'evaluations')
    
    parser = argparse.ArgumentParser(description='CNP Analysis Tool')
    parser.add_argument('--eval_dir', type=str, default=default_eval_dir,
                        help='Directory containing evaluation data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations')
    parser.add_argument('--multiple', action='store_true',
                        help='Use multiple experiments data integration')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Specific experiment directories to use')
    
    args = parser.parse_args()
    
    success = run_analysis(args.eval_dir, args.output_dir, args.multiple, args.experiments)
    
    if success:
        print("Analysis completed successfully!")
    else:
        print("Analysis failed.")
        sys.exit(1)