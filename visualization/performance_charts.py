import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_performance_charts(results_dir, output_dir="./plots"):
    """create performance charts from the results directory"""
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Warning: metrics file {metrics_file} not found")
        config_file = os.path.join(results_dir, "experiment_config.json")
        if os.path.exists(config_file):
            print(f"using alternatives from {config_file} ")
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                agents_data = []
                for agent in config.get("agents", []):
                    agents_data.append({
                        "Agent": agent,
                        "Win Rate (%)": 0,  #  dummy data
                        "Avg Time (ms)": 0,  
                        "Memory (MB)": 0  
                    })
                
                if agents_data:
                    df = pd.DataFrame(agents_data)
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, f"simulation settings:\n"
                            f"num of episodes: {config.get('episodes', 'N/A')}\n"
                            f"date: {config.get('timestamp', 'N/A')}\n"
                            f"Agent used: {', '.join(config.get('agents', []))}",
                            ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    plt.savefig(os.path.join(output_dir, "simulation_info.png"))
                    plt.close()
                    
                    print(f"saved simulation details to {output_dir} .")
                    return True
            except Exception as e:
                print(f"error in altanative file process: {e}")
                return False
        return False
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error:  {metrics_file} is not a valid JSON file")
        return False
        
    # get agent performance data
    agents_data = []
    try:
        for agent_name, metrics in data.get("agent_performance", {}).items():
            agent_data = {
                "Agent": agent_name,
                "Win Rate (%)": metrics.get("win_contribution", 0) * 100,
                "Avg Time (ms)": metrics.get("avg_time_ms", 0),
                "Memory (MB)": data.get("resource_usage", {}).get(agent_name, {}).get("avg_memory_mb", 0)
            }
            agents_data.append(agent_data)
    except (KeyError, AttributeError) as e:
        print(f"Error: unexpected data types: {e}")
        return False
    
    if not agents_data:
        print("WARN: No agent performance data found in the metrics file")
        return False
        
    df = pd.DataFrame(agents_data)
    
    sns.set_theme(style="whitegrid")
    
    # win rate comparison 
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(x="Agent", y="Win Rate (%)", hue="Agent", data=df, palette="viridis", dodge=False)
    chart.bar_label(chart.containers[0], fmt='%.1f%%')
    plt.title("Agent Win Rate Comparison", fontsize=16)
    plt.savefig(os.path.join(output_dir, "win_rate_comparison.png"))
    
    # time comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Agent", y="Avg Time (ms)", hue="Agent", data=df, palette="rocket", dodge=False)
    plt.title("Agent Response Time Comparison", fontsize=16)
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    
    # trade-off chart
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Avg Time (ms)", y="Win Rate (%)", 
                  hue="Agent", size="Memory (MB)",
                  sizes=(100, 500), data=df)
    plt.title("Performance vs. Resource Trade-off", fontsize=16)
    plt.savefig(os.path.join(output_dir, "tradeoff.png"))
    
    print(f"{len(df)} performance chart saved in {output_dir} ")
    

    return True