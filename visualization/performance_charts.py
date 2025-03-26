import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_performance_charts(results_dir, output_dir="./plots"):
    """エージェント性能の比較チャートを生成"""
    os.makedirs(output_dir, exist_ok=True)
    
    # メトリクスデータのロード
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    # metrics.jsonが見つからない場合は代替ファイルを探す
    if not os.path.exists(metrics_file):
        print(f"Warning: メトリクスファイル {metrics_file} が見つかりません")
        # experiment_config.jsonからメタデータを取得
        config_file = os.path.join(results_dir, "experiment_config.json")
        if os.path.exists(config_file):
            print(f"代替として {config_file} からデータを取得します")
            # experiment_config.jsonを基本的なグラフ表示に使用
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # 基本的なエージェント情報を生成
                agents_data = []
                for agent in config.get("agents", []):
                    agents_data.append({
                        "Agent": agent,
                        "Win Rate (%)": 0,  # ダミーデータ
                        "Avg Time (ms)": 0,  # ダミーデータ
                        "Memory (MB)": 0     # ダミーデータ
                    })
                
                if agents_data:
                    df = pd.DataFrame(agents_data)
                    # 設定情報をメモとして残す
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, f"シミュレーション設定:\n"
                            f"エピソード数: {config.get('episodes', 'N/A')}\n"
                            f"実行日時: {config.get('timestamp', 'N/A')}\n"
                            f"エージェント: {', '.join(config.get('agents', []))}",
                            ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    plt.savefig(os.path.join(output_dir, "simulation_info.png"))
                    plt.close()
                    
                    print(f"シミュレーション設定情報を {output_dir} に保存しました")
                    return True
            except Exception as e:
                print(f"代替ファイル処理中にエラー: {e}")
                return False
        return False
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: メトリクスファイル {metrics_file} の形式が不正です")
        return False
        
    # DataFrame作成
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
        print(f"Error: メトリクスデータの構造が予期しない形式です: {e}")
        return False
    
    # データが空なら処理終了
    if not agents_data:
        print("警告: 有効なエージェントデータがありません")
        return False
        
    df = pd.DataFrame(agents_data)
    
    # テーマ設定
    sns.set_theme(style="whitegrid")
    
    # 勝率比較チャート - ここを修正
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(x="Agent", y="Win Rate (%)", data=df, palette="viridis")
    if not df.empty:
        chart.bar_label(chart.containers[0], fmt='%.1f%%')
    plt.title("Agent Win Rate Comparison", fontsize=16)
    plt.savefig(os.path.join(output_dir, "win_rate_comparison.png"))
    
    # 計算時間比較 - ここも修正
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Agent", y="Avg Time (ms)", data=df, palette="rocket")
    plt.title("Agent Response Time Comparison", fontsize=16)
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    
    # 性能とリソースのトレードオフ
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Avg Time (ms)", y="Win Rate (%)", 
                  hue="Agent", size="Memory (MB)",
                  sizes=(100, 500), data=df)
    plt.title("Performance vs. Resource Trade-off", fontsize=16)
    plt.savefig(os.path.join(output_dir, "tradeoff.png"))
    
    print(f"{len(df)} エージェントのパフォーマンスチャートを {output_dir} に保存しました")
    
    # 成功を示す値を返す
    return True