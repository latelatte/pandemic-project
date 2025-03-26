import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

def create_performance_charts(results_dir, output_dir="./plots"):
    """複数アルゴリズム間の性能比較チャート"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 結果データ収集
    data = []
    for algorithm in os.listdir(results_dir):
        algo_dir = os.path.join(results_dir, algorithm)
        if not os.path.isdir(algo_dir):
            continue
            
        # 結果JSONファイル読み込み
        result_file = os.path.join(algo_dir, "summary.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
                
            # 基本メトリクスを抽出
            win_rate = result.get("win_rate", 0)
            avg_turns = result.get("avg_turns", 0)
            avg_outbreaks = result.get("avg_outbreaks", 0)
            
            # リソース使用量
            cpu_usage = result.get("resource_usage", {}).get("avg_cpu_percent", 0)
            memory_usage = result.get("resource_usage", {}).get("avg_memory_mb", 0)
            
            # データフレーム用にデータ追加
            data.append({
                "Algorithm": algorithm,
                "Win Rate (%)": win_rate * 100,
                "Avg. Turns": avg_turns,
                "Avg. Outbreaks": avg_outbreaks,
                "CPU Usage (%)": cpu_usage,
                "Memory (MB)": memory_usage
            })
    
    # データフレーム作成
    df = pd.DataFrame(data)
    
    # テーマ設定
    sns.set_theme(style="whitegrid")
    
    # 1. 勝率比較バーチャート
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(x="Algorithm", y="Win Rate (%)", data=df, palette="viridis")
    chart.bar_label(chart.containers[0], fmt='%.1f%%')
    plt.title("アルゴリズム別勝率比較", fontsize=16)
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, "win_rate_comparison.png"), dpi=300)
    plt.close()
    
    # 2. リソース使用量と性能のバブルチャート
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x="CPU Usage (%)", 
        y="Memory (MB)",
        size="Win Rate (%)",
        sizes=(100, 2000),
        hue="Algorithm",
        data=df
    )
    
    # アノテーション追加
    for i, row in df.iterrows():
        plt.annotate(
            row["Algorithm"],
            (row["CPU Usage (%)"], row["Memory (MB)"]),
            fontsize=9
        )
        
    plt.title("リソース使用量と勝率の関係", fontsize=16)
    plt.savefig(os.path.join(output_dir, "resource_performance_bubble.png"), dpi=300)
    plt.close()
    
    # 3. ターン数・アウトブレイク数のペアプロット
    metrics_df = df[["Algorithm", "Win Rate (%)", "Avg. Turns", "Avg. Outbreaks"]]
    sns.pairplot(metrics_df, hue="Algorithm")
    plt.savefig(os.path.join(output_dir, "metrics_pairplot.png"), dpi=300)
    plt.close()