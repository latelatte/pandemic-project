import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob

def load_agent_data(results_dir):
    """メトリクスファイルからエージェントデータをロード"""
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Warning: メトリクスファイル {metrics_file} が見つかりません")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        agents_data = []
        for agent_name, metrics in data.get("agent_performance", {}).items():
            resource_usage = data.get("resource_usage", {}).get(agent_name, {})
            memory_mb = resource_usage.get("avg_memory_mb", 0)
            cpu_percent = resource_usage.get("avg_cpu_percent", 0)
            
            win_rate = metrics.get("win_contribution", 0) * 100
            avg_time_ms = metrics.get("avg_time_ms", 0)
            
            # CNP計算
            cnp = calculate_cnp(win_rate, memory_mb, avg_time_ms)
            
            # 追加のコスト効率指標
            time_efficiency = win_rate / max(1, avg_time_ms) if avg_time_ms > 0 else 0
            memory_efficiency = win_rate / max(1, memory_mb) if memory_mb > 0 else 0
            
            agent_data = {
                "Agent": agent_name,
                "Win Rate (%)": win_rate,
                "Avg Time (ms)": avg_time_ms,
                "Memory (MB)": memory_mb,
                "CPU (%)": cpu_percent,
                "CNP": cnp,
                "Time Efficiency": time_efficiency,
                "Memory Efficiency": memory_efficiency
            }
            agents_data.append(agent_data)
            
        return pd.DataFrame(agents_data) if agents_data else None
        
    except Exception as e:
        print(f"データ読み込み中にエラー: {e}")
        return None

def aggregate_experiment_data(results_dir="./logs", n_latest=5, pattern="experiment_*"):
    """複数の実験結果を統合して統計的に信頼性の高いデータを生成"""
    # 実験ディレクトリを探す
    if os.path.isdir(results_dir) and not results_dir.endswith("logs"):
        # 単一の実験ディレクトリが指定された場合
        experiment_dirs = [results_dir]
    else:
        # logsディレクトリから最新のn_latest個の実験を見つける
        base_dir = results_dir if results_dir.endswith("logs") else "./logs"
        experiment_dirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                               if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("experiment_")])
        experiment_dirs = experiment_dirs[-n_latest:] if len(experiment_dirs) >= n_latest else experiment_dirs
    
    # 全実験から集めたデータ
    all_agent_data = {}
    
    for exp_dir in experiment_dirs:
        metrics_file = os.path.join(exp_dir, "metrics.json")
        if not os.path.exists(metrics_file):
            print(f"警告: {metrics_file} が見つかりません。スキップします。")
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
            print(f"エラー: {metrics_file} の読み込み中に問題が発生しました: {e}")
    
    # 集約データをDataFrameに変換
    aggregated_data = []
    for agent_name, data in all_agent_data.items():
        if not data["win_rates"]:  # データがない場合はスキップ
            continue
            
        # CNP計算のための値
        win_rate = np.mean(data["win_rates"])
        avg_time = np.mean(data["times"])
        memory = np.mean(data["memory"])
        
        # 各種効率性指標の計算
        cnp = calculate_cnp(win_rate, memory, avg_time)
        time_efficiency = win_rate / max(1, avg_time/100) if avg_time > 0 else 0
        memory_efficiency = win_rate / max(1, memory/100) if memory > 0 else 0
        
        # CNPの標準偏差を計算（伝播誤差の法則を使用）
        cnp_values = []
        for w, t, m in zip(data["win_rates"], data["times"], data["memory"]):
            if m > 0 and t > 0:
                cnp_val = w / ((m/1000) * (t/3600000))
                cnp_values.append(cnp_val)
        
        cnp_std = np.std(cnp_values) if cnp_values else 0
        
        aggregated_data.append({
            "Agent": agent_name,
            "Win Rate (%)": win_rate,
            "Win Rate StdDev": np.std(data["win_rates"]),
            "Avg Time (ms)": avg_time,
            "Time StdDev": np.std(data["times"]),
            "Memory (MB)": memory,
            "Memory StdDev": np.std(data["memory"]),
            "CPU (%)": np.mean(data["cpu"]),
            "CPU StdDev": np.std(data["cpu"]),
            "CNP": cnp,
            "CNP StdDev": cnp_std,
            "Time Efficiency": time_efficiency, 
            "Memory Efficiency": memory_efficiency,
            "n_samples": len(data["win_rates"])
        })
    
    return pd.DataFrame(aggregated_data) if aggregated_data else None

def create_cnp_chart(df, output_dir):
    """コスト正規化性能チャートを作成（エラーバー付き）"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # バープロット
    bars = plt.bar(df["Agent"], df["CNP"], yerr=df["CNP StdDev"], 
              capsize=10, color='skyblue', 
              error_kw={'ecolor': 'gray', 'capthick': 2})
    
    # バーの上に値を表示
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title("Cost-Normalized Performance (CNP)", fontsize=16)
    plt.xlabel("Agent", fontsize=13)
    plt.ylabel("CNP Score (win rate / (Memory × Time))", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # CNPの説明を図に追加
    plt.figtext(0.1, 0.01, 
               "CNP: Cost-normalised performance indicators - efficiency indicators that show how much higher the win rate is for the same resource consumption.", 
               wrap=True, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # テキスト用に下部スペース確保
    plt.savefig(os.path.join(output_dir, "cost_normalized_performance.png"), dpi=300)
    plt.close()

def create_efficiency_comparison(df, output_dir):
    """時間効率とメモリ効率の比較図を作成（エラーバー付き）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 時間効率チャート
    bars1 = ax1.bar(df["Agent"], df["Time Efficiency"], color='tomato')
    ax1.set_title("Time Efficiency (Win Rate / Time)", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylabel("Efficiency Score", fontsize=12)
    
    # バーの上に値を表示
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # メモリ効率チャート
    bars2 = ax2.bar(df["Agent"], df["Memory Efficiency"], color='mediumseagreen')
    ax2.set_title("Memory Efficiency (Win Rate / Memory)", fontsize=14) 
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylabel("Efficiency Score", fontsize=12)
    
    # バーの上に値を表示
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_comparison.png"), dpi=300)
    plt.close()

def create_resource_usage_chart(df, output_dir):
    """リソース使用量の散布図を作成（勝率をカラーマップで表示）"""
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(
        df["Avg Time (ms)"], 
        df["Memory (MB)"],
        c=df["Win Rate (%)"],
        s=200,
        cmap="viridis",
        alpha=0.7
    )
    
    # エラーバーの追加
    for i, row in df.iterrows():
        plt.errorbar(
            row["Avg Time (ms)"], row["Memory (MB)"],
            xerr=row["Time StdDev"], yerr=row["Memory StdDev"],
            fmt='none', ecolor='gray', alpha=0.5
        )
    
    # 各点にエージェント名のラベルを追加
    for i, agent in enumerate(df["Agent"]):
        plt.annotate(
            agent,
            (df["Avg Time (ms)"].iloc[i], df["Memory (MB)"].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10
        )
    
    cbar = plt.colorbar(scatter, label="Win Rate (%)")
    
    # CNP値を散布図にテキスト表示
    for i, row in df.iterrows():
        plt.annotate(
            f"CNP: {row['CNP']:.2f}",
            (row["Avg Time (ms)"], row["Memory (MB)"]),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=8
        )
    
    plt.xlabel("Average Response Time (ms)", fontsize=12)
    plt.ylabel("Memory Usage (MB)", fontsize=12)
    plt.title("Resource Usage vs Performance", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 効率の等高線（理論的なCNP値）を描画
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    
    # 理論上のCNP等高線を描画
    x = np.linspace(x_min, x_max, 100)
    
    cnp_contours = [0.1, 0.5, 1, 5, 10, 50]
    for cnp in cnp_contours:
        # CNP = win_rate / (memory * time)
        # この式をwinrate = CNP * memory * timeに変形
        # 勝率100%と仮定すると、memory = 100 / (CNP * time)
        y = 100 / (cnp * x/3600000)  # GB-hrs単位に揃える
        
        # プロット範囲内のみ描画
        valid_points = (y >= y_min) & (y <= y_max)
        if np.any(valid_points):
            plt.plot(x[valid_points], y[valid_points], '--', color='darkblue', alpha=0.3)
            # 曲線にラベルを付ける
            mid_idx = np.where(valid_points)[0][len(np.where(valid_points)[0])//2]
            plt.text(x[mid_idx], y[mid_idx], f"CNP={cnp}", color='darkblue', alpha=0.7, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_usage_performance.png"), dpi=300)
    plt.close()

def create_statistical_comparison(df, output_dir):
    """統計的有意性を含むエージェント比較グラフ"""
    plt.figure(figsize=(12, 8))
    
    # 勝率の比較（エラーバー付き）
    x = np.arange(len(df))
    width = 0.25
    
    # 勝率バー
    plt.bar(x - width, df["Win Rate (%)"], width, 
           yerr=df["Win Rate StdDev"], 
           label="Win Rate (%)", color='cornflowerblue',
           capsize=5)
    
    # 時間バー (スケール調整)
    time_scale = df["Win Rate (%)"].max() / df["Avg Time (ms)"].max()
    plt.bar(x, df["Avg Time (ms)"] * time_scale, width,
           yerr=df["Time StdDev"] * time_scale,
           label=f"Time (ms) × {time_scale:.5f}", color='lightcoral',
           capsize=5)
    
    # メモリバー (スケール調整)
    mem_scale = df["Win Rate (%)"].max() / df["Memory (MB)"].max()
    plt.bar(x + width, df["Memory (MB)"] * mem_scale, width,
           yerr=df["Memory StdDev"] * mem_scale,
           label=f"Memory (MB) × {mem_scale:.5f}", color='mediumseagreen',
           capsize=5)
    
    plt.xlabel("Agent", fontsize=13)
    plt.ylabel("Scaled Metrics", fontsize=13)
    plt.title("Statistical Comparison of Agent Performance", fontsize=16)
    plt.xticks(x, df["Agent"])
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend()
    
    # CNP値をテキスト表示
    for i, row in df.iterrows():
        plt.text(i, 5, f"CNP: {row['CNP']:.2f} ± {row['CNP StdDev']:.2f}",
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_comparison.png"), dpi=300)
    plt.close()

def run_cnp_analysis(results_dir, output_dir=None, n_latest=5):
    """コスト効率性分析の実行関数"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(results_dir) if os.path.isdir(results_dir) else results_dir, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 複数実験の集約データを取得
    print(f"最新{n_latest}回の実験データを集約中...")
    df = aggregate_experiment_data(results_dir, n_latest)
    
    if df is None or df.empty:
        print("有効なエージェントデータがありません")
        return False
    
    print(f"集計されたエージェント: {', '.join(df['Agent'].unique())}")
    print(f"サンプル数: {df['n_samples'].iloc[0]}実験")
    
    # CNPチャート作成
    print("コスト正規化性能(CNP)チャートを作成中...")
    create_cnp_chart(df, output_dir)
    
    # 効率性比較チャート作成
    print("効率性比較チャートを作成中...")
    create_efficiency_comparison(df, output_dir)
    
    # リソース使用量チャート作成
    print("リソース使用量パフォーマンスマップを作成中...")
    create_resource_usage_chart(df, output_dir)
    
    # 統計比較チャート作成
    print("統計的比較チャートを作成中...")
    create_statistical_comparison(df, output_dir)
    
    print(f"コスト効率性分析が完了しました: {output_dir}")
    return True

def calculate_cnp(win_rate, memory_mb, time_ms):
    # 単位変換
    memory_gb = memory_mb / 1024.0
    time_hrs = time_ms / (3600 * 1000)
    
    # ゼロ除算防止
    if memory_gb <= 0 or time_hrs <= 0:
        return 0
    
    return win_rate / (memory_gb * time_hrs)

if __name__ == "__main__":
    import sys
    
    # 最新の実験ディレクトリを探す
    def find_latest_experiment():
        log_dirs = sorted([d for d in os.listdir("./logs") if d.startswith("experiment_")])
        if not log_dirs:
            return None
        return os.path.join("./logs", log_dirs[-1])
    
    # コマンドライン引数の解析
    import argparse
    parser = argparse.ArgumentParser(description='コスト正規化性能(CNP)分析ツール')
    parser.add_argument('--dir', type=str, default="../logs", help='分析する実験ディレクトリ (デフォルト: 最新)')
    parser.add_argument('--n_latest', type=int, default=5, help='集計する実験数 (デフォルト: 5)')
    parser.add_argument('--output', type=str, default="./plots", help='出力ディレクトリ (デフォルト: <実験DIR>/plots)')
    
    args = parser.parse_args()
    results_dir = args.dir if args.dir else find_latest_experiment()
    
    if not results_dir:
        print("実験ディレクトリが見つかりません。")
        sys.exit(1)
    
    print(f"分析対象: {results_dir}")
    success = run_cnp_analysis(results_dir, args.output, args.n_latest)
    print("分析完了" if success else "分析失敗")