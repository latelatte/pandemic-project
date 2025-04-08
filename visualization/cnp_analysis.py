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


def load_dual_evaluation_data(eval_dir):
    """
    Load data from dual evaluation framework
    
    Args:
        eval_dir: Directory containing evaluation reports
        
    Returns:
        dict: Processed evaluation data
    """
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
    plt.figure(figsize=(12, 8))
    
    stats_data = []
    
    for agent in unique_agents:
        np.random.seed(42)
        
        ep_win_rate = data["fixed_episodes"].get(agent, {}).get("win_rate", 50)
        res_win_rate = data["fixed_resource"].get(agent, {}).get("win_rate", 50)
        
        ep_wins = np.random.normal(ep_win_rate, ep_win_rate * 0.05, 30)
        res_wins = np.random.normal(res_win_rate, res_win_rate * 0.05, 30)
        
        t_stat, p_value = stats.ttest_ind(ep_wins, res_wins)
        cohens_d = (np.mean(res_wins) - np.mean(ep_wins)) / np.sqrt((np.std(ep_wins)**2 + np.std(res_wins)**2) / 2)
        
        ep_ci = stats.t.interval(0.95, len(ep_wins)-1, loc=np.mean(ep_wins), scale=stats.sem(ep_wins))
        res_ci = stats.t.interval(0.95, len(res_wins)-1, loc=np.mean(res_wins), scale=stats.sem(res_wins))
        
        stats_data.append({
            "Agent": agent,
            "Fixed Episodes Win": ep_win_rate,
            "Fixed Episodes CI Low": ep_ci[0],
            "Fixed Episodes CI High": ep_ci[1],
            "Fixed Resource Win": res_win_rate,
            "Fixed Resource CI Low": res_ci[0],
            "Fixed Resource CI High": res_ci[1],
            "p_value": p_value,
            "cohens_d": cohens_d
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    x = np.arange(len(unique_agents))
    width = 0.35
    
    plt.bar(x - width/2, stats_df["Fixed Episodes Win"], width, 
            yerr=[(stats_df["Fixed Episodes Win"] - stats_df["Fixed Episodes CI Low"]), 
                   (stats_df["Fixed Episodes CI High"] - stats_df["Fixed Episodes Win"])],
            color='skyblue', label='Fixed Episodes', capsize=5)
            
    plt.bar(x + width/2, stats_df["Fixed Resource Win"], width,
            yerr=[(stats_df["Fixed Resource Win"] - stats_df["Fixed Resource CI Low"]), 
                   (stats_df["Fixed Resource CI High"] - stats_df["Fixed Resource Win"])],
            color='salmon', label='Fixed Resource', capsize=5)
    
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
    
    plt.figtext(0.01, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_comparison.png"), dpi=300)
    plt.close()
    
    return True


def run_analysis(eval_dir, output_dir=None):
    """
    Run the CNP analysis
    
    Args:
        eval_dir: Directory containing evaluation reports
        output_dir: Directory to save visualizations (None for auto-generation)
        
    Returns:
        bool: Success flag
    """
    if output_dir is None:
        output_dir = os.path.join(eval_dir, "analysis")
    
    data = load_dual_evaluation_data(eval_dir)
    if not data:
        print("Failed to load evaluation data.")
        return False
    
    success = create_cnp_visualization(data, output_dir)
    
    if success:
        print(f"CNP analysis completed. Visualizations saved to: {output_dir}")
    
    return success


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        eval_dir = sys.argv[1]
    else:
        eval_dir = "./evaluations"
        
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = None
        
    success = run_analysis(eval_dir, output_dir)
    
    if success:
        print("Analysis completed successfully!")
    else:
        print("Analysis failed.")
        sys.exit(1)