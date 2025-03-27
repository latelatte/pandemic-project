import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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
    """パレートフロンティアを計算
    points: [(x, y)] の形式のリスト。xは最小化、yは最大化したい値。
    """
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
    """2次元パレート分析チャート作成の汎用関数"""
    plt.figure(figsize=(12, 8))
    
    if size_col:
        # サイズ変数がある場合は散布図にサイズ情報を含める
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
    
    # エラーバーの追加
    for i, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        x_err = row.get(f"{x_col.split()[0]} StdDev", 0)
        y_err = row.get(f"{y_col.split()[0]} StdDev", 0)
        
        plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', ecolor='gray', alpha=0.5)
    
    # パレートフロンティア計算
    points = [(row[x_col], row[y_col]) for _, row in df.iterrows()]
    
    # 最適化の方向を調整
    if not minimize_x:
        points = [(-x, y) for x, y in points]
    if not maximize_y:
        points = [(x, -y) for x, y in points]
    
    pareto_points = compute_pareto_frontier(points)
    
    # 方向を元に戻す
    if not minimize_x:
        pareto_points = [(-x, y) for x, y in pareto_points]
    if not maximize_y:
        pareto_points = [(x, -y) for x, y in pareto_points]
    
    # パレートフロンティア線を描画
    if len(pareto_points) >= 2:
        pareto_x, pareto_y = zip(*pareto_points)
        plt.plot(pareto_x, pareto_y, 'k--', label='Pareto Frontier')
    
    # 各点にエージェント名のラベルを追加
    for i, agent in enumerate(df["Agent"]):
        plt.annotate(
            agent,
            (df[x_col].iloc[i], df[y_col].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )
    
    # 最適化方向の矢印を追加
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
    
    # 最適領域にシェーディング
    if len(pareto_points) >= 2:
        pareto_x, pareto_y = zip(*pareto_points)
        if minimize_x and maximize_y:
            # 左上が理想的な領域
            plt.fill_between(pareto_x, pareto_y, [max_y] * len(pareto_x), 
                            alpha=0.1, color='green', label='_ideal_region')
            
            # 理想的な方向を示す矢印
            plt.annotate('Desireble', xy=(min_x + (max_x - min_x) * 0.2, max_y - (max_y - min_y) * 0.2),
                        xytext=(min_x + (max_x - min_x) * 0.3, max_y - (max_y - min_y) * 0.3),
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.5),
                        color='green', alpha=0.7, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def create_2d_pareto_analysis(df, output_dir):
    """2次元パレート分析チャートを作成"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 時間 vs 勝率のパレート分析
    create_2d_pareto_chart(
        df, 
        "Avg Time (ms)", "Win Rate (%)",
        "Average Response Time (ms) - The smaller the better", "Win Rate (%) - The higher the better",
        "Performance vs Win Rate Trade-off Analysis",
        "pareto_time_winrate.png",
        output_dir,
        "Memory (MB)"
    )
    
    # メモリ vs 勝率のパレート分析
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
    """時間、メモリ、勝率の3次元可視化"""
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
        
        # 3D空間でのエラーバー
        for j, row in agent_data.iterrows():
            x = row['Avg Time (ms)']
            y = row['Memory (MB)']
            z = row['Win Rate (%)']
            dx = row.get('Time StdDev', 0)
            dy = row.get('Memory StdDev', 0)
            dz = row.get('Win Rate StdDev', 0)
            
            # X方向のエラーバー
            ax.plot([x-dx, x+dx], [y, y], [z, z], color='gray', alpha=0.3)
            
            # Y方向のエラーバー
            ax.plot([x, x], [y-dy, y+dy], [z, z], color='gray', alpha=0.3)
            
            # Z方向のエラーバー
            ax.plot([x, x], [y, y], [z-dz, z+dz], color='gray', alpha=0.3)
    
    ax.set_xlabel('Average Response Time (ms)', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_zlabel('Win Rate (%)', fontsize=12)
    
    # 最適方向を示す矢印
    max_x = max(df['Avg Time (ms)'].max(), 1)
    max_y = max(df['Memory (MB)'].max(), 1)
    max_z = max(df['Win Rate (%)'].max(), 1)
    
    # X軸(時間)最小化の矢印
    ax.quiver(max_x * 0.9, 0, 0, -max_x * 0.2, 0, 0, color='r', arrow_length_ratio=0.1, label='Minimize')
    ax.text(max_x * 0.8, 0, -max_z * 0.1, "Time ↓", color='r')
    
    # Y軸(メモリ)最小化の矢印
    ax.quiver(0, max_y * 0.9, 0, 0, -max_y * 0.2, 0, color='r', arrow_length_ratio=0.1)
    ax.text(0, max_y * 0.8, -max_z * 0.1, "Memory ↓", color='r')
    
    # Z軸(勝率)最大化の矢印
    ax.quiver(0, 0, max_z * 0.7, 0, 0, max_z * 0.2, color='g', arrow_length_ratio=0.1)
    ax.text(0, -max_y * 0.1, max_z * 0.8, "Win Rate ↑", color='g')
    
    ax.view_init(elev=30, azim=45)
    plt.title("3D Performance Analysis with Error Bars", fontsize=16)
    plt.legend(title="Agent", fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3d_performance_analysis.png"), dpi=300)
    plt.close()

def create_radar_chart(df, output_dir):
    """各エージェントの総合評価をレーダーチャートで可視化"""
    # 正規化
    df_norm = df.copy()
    
    # 評価指標のリスト（最大化したいものと最小化したいもの）
    maximize_metrics = ["Win Rate (%)"]
    minimize_metrics = ["Avg Time (ms)", "Memory (MB)"]
    
    # 正規化処理
    for col in maximize_metrics + minimize_metrics:
        max_val = df[col].max()
        min_val = df[col].min()
        range_val = max_val - min_val
        
        if range_val > 0:
            if col in minimize_metrics:
                # 小さい方が良い指標は反転（1に近いほど良い）
                df_norm[col] = 1 - ((df[col] - min_val) / range_val)
            else:
                # 大きい方が良い指標はそのまま正規化（1に近いほど良い）
                df_norm[col] = (df[col] - min_val) / range_val
        else:
            df_norm[col] = 0.5  # 全て同じ値の場合は中間値に設定
    
    # 表示用のカテゴリ名
    categories = ['Win Rate', 'Time Efficiency', 'Memory Efficiency']
    
    # レーダーチャートの作成
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # カテゴリ数
    N = len(categories)
    
    # 角度の設定
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 閉じた図形にするために最初の要素を追加
    
    # プロット線と塗りつぶし
    for i, agent in enumerate(df_norm['Agent'].unique()):
        agent_data = df_norm[df_norm['Agent'] == agent]
        values = agent_data[["Win Rate (%)", "Avg Time (ms)", "Memory (MB)"]].values.flatten().tolist()
        values += values[:1]  # リストを閉じる
        
        ax.plot(angles, values, linewidth=2, linestyle='-', label=agent)
        ax.fill(angles, values, alpha=0.1)
    
    # グラフの設定
    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # 説明ラベルの追加
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

def run_pareto_analysis(results_dir, output_dir=None, n_latest=5):
    """パレート最適性分析の実行関数"""
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
    
    # 2次元パレート分析
    print("2次元パレート分析チャートを作成中...")
    create_2d_pareto_analysis(df, output_dir)
    
    # 3D可視化
    print("3次元パレート空間を可視化中...")
    create_3d_visualization(df, output_dir)
    
    # レーダーチャート
    print("レーダーチャートを作成中...")
    create_radar_chart(df, output_dir)
    
    print(f"パレート最適性分析が完了しました: {output_dir}")
    return True

# コマンドラインからの実行サポート
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
    parser = argparse.ArgumentParser(description='パレート最適性分析ツール')
    parser.add_argument('--dir', type=str, default="../logs", help='分析する実験ディレクトリ (デフォルト: 最新)')
    parser.add_argument('--n_latest', type=int, default=5, help='集計する実験数 (デフォルト: 5)')
    parser.add_argument('--output', type=str, default="./plots", help='出力ディレクトリ (デフォルト: <実験DIR>/plots)')
    
    args = parser.parse_args()
    results_dir = args.dir if args.dir else find_latest_experiment()
    
    if not results_dir:
        print("実験ディレクトリが見つかりません。")
        sys.exit(1)
    
    print(f"分析対象: {results_dir}")
    success = run_pareto_analysis(results_dir, args.output, args.n_latest)
    print("分析完了" if success else "分析失敗")