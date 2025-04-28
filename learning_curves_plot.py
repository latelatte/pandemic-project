import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import glob
import numpy as np

TITLE_SIZE = 20
LABEL_SIZE = 22
TICK_SIZE = 20
LEGEND_SIZE = 18
ANNOTATION_SIZE = 18

def load_experiment_data(json_dir="./evaluations"):
    """
    最新の実験データのJSONファイルのみを読み込む
    
    Args:
        json_dir: JSONファイルを含むディレクトリ
        
    Returns:
        DataFrame: 学習曲線用のデータフレーム
    """
    all_data = []
    
    summary_files = glob.glob(os.path.join(json_dir, "**/convergence_summary.json"), recursive=True)
    

    summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    latest_file = summary_files[0]
    print(f"最新の実験データを使用します: {latest_file}")
    
    try:
        with open(latest_file, 'r') as f:
            experiment_data = json.load(f)
        
        if "results" in experiment_data:
            for agent_name, agent_data in experiment_data["results"].items():
                if "learning_curve" in agent_data:
                    learning_curve = agent_data["learning_curve"]
                    episodes = learning_curve.get("episodes", [])
                    win_rates = learning_curve.get("win_rates", [])
                    
                    
                    win_rates_percent = [rate * 100 for rate in win_rates]
                    
                    for i in range(len(episodes)):
                        if i < len(win_rates):
                            all_data.append({
                                'Episode': episodes[i],
                                'Algorithm': agent_name,
                                'Run': '1',
                                'Win Rate (%)': win_rates_percent[i]
                            })
        # else:
        #     print(f"警告: {latest_file} の構造を認識できません")
    
    except Exception as e:
        print(f"error in {latest_file}: {str(e)}")
    
    if all_data:
        return pd.DataFrame(all_data)
    else:
        print("警告: 有効なデータがありません")
        return pd.DataFrame()

def apply_smoothing(df, window_size=30):
    """
    データフレームに移動平均を適用する
    
    Args:
        df: 元のデータフレーム
        window_size: 移動平均のウィンドウサイズ
        
    Returns:
        DataFrame: 平滑化されたデータを含むデータフレーム
    """
    smoothed_df = df.copy()
    
    for algo in df['Algorithm'].unique():
        algo_mask = df['Algorithm'] == algo
        for run in df.loc[algo_mask, 'Run'].unique():
            mask = (df['Algorithm'] == algo) & (df['Run'] == run)
            
            temp_df = df[mask].sort_values('Episode')
            smoothed_df.loc[mask, 'Win Rate (%) Smoothed'] = temp_df['Win Rate (%)'].rolling(
                window=window_size, min_periods=1, center=True).mean()

            smoothed_df.loc[mask, 'Win Rate (%) Std'] = temp_df['Win Rate (%)'].rolling(
                window=window_size, min_periods=1).std()
    
    return smoothed_df

def calculate_confidence_intervals(df):
    """
    95%信頼区間を計算
    
    Args:
        df: 標準偏差を含むデータフレーム
        
    Returns:
        DataFrame: 信頼区間を追加したデータフレーム
    """
    for algo in df['Algorithm'].unique():
        for run in df[df['Algorithm'] == algo]['Run'].unique():
            mask = (df['Algorithm'] == algo) & (df['Run'] == run)
            
            valid_points = df.loc[mask, 'Win Rate (%) Std'].notna().sum()
            if valid_points > 0:
                n = max(30, valid_points / 10)
                df.loc[mask, 'CI Upper'] = df.loc[mask, 'Win Rate (%) Smoothed'] + 1.96 * df.loc[mask, 'Win Rate (%) Std'] / np.sqrt(n)
                df.loc[mask, 'CI Lower'] = df.loc[mask, 'Win Rate (%) Smoothed'] - 1.96 * df.loc[mask, 'Win Rate (%) Std'] / np.sqrt(n)
                df.loc[mask, 'CI Lower'] = df.loc[mask, 'CI Lower'].clip(lower=0)
    
    return df

def create_learning_curves_plot(df, output_dir='./evaluations/analysis', skip_initial_episodes=100, max_y_limit=2.0):
    """
    学習曲線の改良版プロットを作成する
    
    Args:
        df: 学習曲線データを含むDataFrame
        output_dir: 出力ディレクトリ (デフォルト: ./evaluations/analysis)
        skip_initial_episodes: 初期のエピソード数をスキップする (デフォルト: 100)
        max_y_limit: Y軸の最大値 (デフォルト: 2.0)
    """

    
    smoothed_df = apply_smoothing(df, window_size=30)
    
    smoothed_df = calculate_confidence_intervals(smoothed_df)
    if skip_initial_episodes > 0:
        min_episodes = {}
        for algo in smoothed_df['Algorithm'].unique():
            min_ep = smoothed_df[smoothed_df['Algorithm'] == algo]['Episode'].min()
            min_episodes[algo] = min_ep + skip_initial_episodes

        filtered_df = pd.DataFrame()
        for algo in smoothed_df['Algorithm'].unique():
            algo_data = smoothed_df[smoothed_df['Algorithm'] == algo]
            filtered_data = algo_data[algo_data['Episode'] >= min_episodes[algo]]
            filtered_df = pd.concat([filtered_df, filtered_data])
        
        if not filtered_df.empty:
            smoothed_df = filtered_df
            print(f"最初の {skip_initial_episodes} エピソードをスキップしました")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    algorithms = smoothed_df['Algorithm'].unique()
    
    colors = {
        'MCTSAgent': 'green',
        'EAAgent': 'blue', 
        'MARLAgent': 'red',
        'MCTS': 'green',
        'EA': 'blue', 
        'MARL': 'red',
        'PPO': 'purple',
        'DQN': 'orange',
        'A2C': 'brown',
        'SAC': 'pink'
    }
    
    markers = {
        'MCTSAgent': '^', 'MCTS': '^',
        'EAAgent': 's', 'EA': 's',
        'MARLAgent': 'o', 'MARL': 'o',
        'PPO': 'D', 'DQN': '*', 'A2C': 'X', 'SAC': 'P'
    }
    
    linestyles = {
        'MCTSAgent': '-', 'MCTS': '-',
        'EAAgent': '--', 'EA': '--',
        'MARLAgent': ':', 'MARL': ':',
        'PPO': '-.', 'DQN': '-', 'A2C': '--', 'SAC': ':'
    }
    
    marker_step = max(1, len(smoothed_df) // 20)
    
    for algo in algorithms:
        display_name = algo.replace('Agent', '')
        
        algo_data = smoothed_df[smoothed_df['Algorithm'] == algo]
        
        color = colors.get(algo, sns.color_palette("husl", len(algorithms))[list(algorithms).index(algo)])
        marker = markers.get(algo, 'o')
        linestyle = linestyles.get(algo, '-')

        if 'CI Upper' in algo_data.columns and 'CI Lower' in algo_data.columns:
            plt.fill_between(
                algo_data['Episode'], 
                algo_data['CI Lower'], 
                algo_data['CI Upper'], 
                alpha=0.2, 
                color=color
            )
        
        plt.plot(
            algo_data['Episode'], 
            algo_data['Win Rate (%) Smoothed'], 
            label=display_name, 
            color=color, 
            linestyle=linestyle, 
            linewidth=2
        )
        
        plt.plot(
            algo_data.iloc[::marker_step]['Episode'], 
            algo_data.iloc[::marker_step]['Win Rate (%) Smoothed'], 
            color=color, 
            marker=marker, 
            linestyle='none', 
            markersize=8
        )
    
    final_annotations = []
    for algo in algorithms:
        algo_data = smoothed_df[smoothed_df['Algorithm'] == algo].sort_values('Episode')
        if not algo_data.empty:
            final_idx = algo_data['Episode'].idxmax()
            final_point = algo_data.loc[final_idx]
            final_ep = final_point['Episode']
            final_rate = final_point['Win Rate (%) Smoothed']
            
            if 'MCTS' in algo:
                label = f"MCTS\nFinal: {final_rate:.2f}%"
                xytext = (final_ep - 200, final_rate + 0.2)
            elif 'EA' in algo:
                label = f"EA\nFinal: {final_rate:.2f}%"
                xytext = (final_ep - 200, final_rate + 0.35)
            elif 'MARL' in algo:
                label = f"MARL\nFinal: {final_rate:.2f}%"
                xytext = (final_ep - 150, final_rate + 0.2)
            else:
                label = f"{algo.replace('Agent', '')}\nFinal: {final_rate:.2f}%"
                xytext = (final_ep - 150, final_rate + 0.1)
            
            final_annotations.append({
                'text': label,
                'xy': (final_ep, final_rate),
                'xytext': xytext
            })
            
    feature_annotations = []
    for algo in algorithms:
        algo_data = smoothed_df[smoothed_df['Algorithm'] == algo].sort_values('Episode')
        
        if len(algo_data) >= 20:
            algo_data['slope'] = algo_data['Win Rate (%) Smoothed'].diff() / algo_data['Episode'].diff().clip(lower=1)
            
            if 'EA' in algo:
                plateaus = algo_data[abs(algo_data['slope']) < 0.0005]
                if not plateaus.empty and len(plateaus) > 5:
                    middle_idx = plateaus.iloc[len(plateaus) // 2].name
                    plateau_point = algo_data.loc[middle_idx]
                    feature_annotations.append({
                        'text': "EA Plateau",
                        'xy': (plateau_point['Episode'], plateau_point['Win Rate (%) Smoothed']),
                        'xytext': (plateau_point['Episode'] - 150, plateau_point['Win Rate (%) Smoothed'] + 0.60)
                    })
            
            if 'MARL' in algo:
                second_half = algo_data.iloc[len(algo_data) // 2:]
                negative_slopes = second_half[second_half['slope'] < -0.0002]
                if not negative_slopes.empty and len(negative_slopes) > 3:
                    decline_idx = negative_slopes['slope'].idxmin()
                    decline_point = algo_data.loc[decline_idx]
                    feature_annotations.append({
                        'text': "MARL\nPerformance\nDegradation",
                        'xy': (decline_point['Episode'], decline_point['Win Rate (%) Smoothed']),
                        'xytext': (decline_point['Episode'] - 200, decline_point['Win Rate (%) Smoothed'] + 0)
                    })
    
    for anno in final_annotations:
        plt.annotate(
            anno['text'], 
            xy=anno['xy'], 
            xytext=anno['xytext'],
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=14
        )
    
    for anno in feature_annotations:
        plt.annotate(
            anno['text'], 
            xy=anno['xy'], 
            xytext=anno['xytext'],
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=16
        )
    
    plt.title('Learning Curves: Win Rate Comparison', fontsize=20)
    plt.xlabel('Training Episodes', fontsize=18)
    plt.ylabel('Win Rate (%)', fontsize=18)
    

    plt.ylim(0, max_y_limit)
    
    min_x = smoothed_df['Episode'].min()
    max_x = smoothed_df['Episode'].max()
    plt.xlim(min_x, max_x)
    
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # if skip_initial_episodes > 0:
    #     plt.figtext(
    #         0.5, 0.01,
    #         f"Note: Initial {skip_initial_episodes} episodes omitted. High win rates in early episodes reflect statistical variance due to small sample sizes.",
    #         ha="center", fontsize=16, style='italic'
    #     )
    
    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, 'learning_curves')

    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    
    print(f"学習曲線を {output_base}.png に保存しました")
    
    plt.show()

def main():
    """
    メイン関数 - 引数なしで実行
    """
    # 固定パスからデータを読み込む
    df = load_experiment_data("./evaluations")
    
    if not df.empty:
        # データの前処理（異常に高い値をチェック）
        max_win_rate = df['Win Rate (%)'].max()
        if max_win_rate > 100:
            print(f"警告: データに異常に高い勝率があります（最大値: {max_win_rate}%）。データを確認してください。")
        
        # 学習曲線をプロット（初期の100エピソードを除外、Y軸最大値を2.0%に設定）
        create_learning_curves_plot(df, skip_initial_episodes=100, max_y_limit=2.0)


if __name__ == "__main__":
    main()