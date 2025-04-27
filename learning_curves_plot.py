import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import glob
import numpy as np

def load_experiment_data(json_dir="./evaluations"):
    """
    最新の実験データのJSONファイルのみを読み込む
    
    Args:
        json_dir: JSONファイルを含むディレクトリ
        
    Returns:
        DataFrame: 学習曲線用のデータフレーム
    """
    all_data = []
    
    # convergence_summary.jsonファイルを探す
    summary_files = glob.glob(os.path.join(json_dir, "**/convergence_summary.json"), recursive=True)
    
    if not summary_files:
        print(f"警告: {json_dir} にconvergence_summary.jsonファイルが見つかりませんでした")
        return pd.DataFrame()
    
    # ファイルの最終更新日時でソート
    summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 最新のファイルのみを使用
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
                    
                    print(f"エージェント {agent_name} の学習曲線データ: {len(episodes)} エピソード")
                    
                    # win_ratesを割合（0-1）から百分率（0-100）に変換
                    win_rates_percent = [rate * 100 for rate in win_rates]
                    
                    for i in range(len(episodes)):
                        if i < len(win_rates):
                            all_data.append({
                                'Episode': episodes[i],
                                'Algorithm': agent_name,
                                'Run': '1',
                                'Win Rate (%)': win_rates_percent[i]
                            })
        else:
            print(f"警告: {latest_file} の構造を認識できません")
    
    except Exception as e:
        print(f"エラー: {latest_file} の読み込み中に問題が発生しました: {str(e)}")
    
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
    
    # アルゴリズムとラン別にグループ化して移動平均を適用
    for algo in df['Algorithm'].unique():
        algo_mask = df['Algorithm'] == algo
        for run in df.loc[algo_mask, 'Run'].unique():
            mask = (df['Algorithm'] == algo) & (df['Run'] == run)
            
            # ソートしてから移動平均を適用
            temp_df = df[mask].sort_values('Episode')
            smoothed_df.loc[mask, 'Win Rate (%) Smoothed'] = temp_df['Win Rate (%)'].rolling(
                window=window_size, min_periods=1, center=True).mean()
                
            # 信頼区間用の標準偏差も計算
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
    # 各アルゴリズムとランごとの移動平均ウィンドウサイズを推定
    for algo in df['Algorithm'].unique():
        for run in df[df['Algorithm'] == algo]['Run'].unique():
            mask = (df['Algorithm'] == algo) & (df['Run'] == run)
            
            # NaNを除外して有効なデータポイント数を取得
            valid_points = df.loc[mask, 'Win Rate (%) Std'].notna().sum()
            if valid_points > 0:
                # 95%信頼区間 = 平均 ± 1.96 * (標準偏差 / √n)
                # ここでnはサンプルサイズ（ウィンドウサイズ）
                # 非常に小さな値を避けるためのクリッピング
                n = max(30, valid_points / 10)  # 経験的な値、調整可能
                df.loc[mask, 'CI Upper'] = df.loc[mask, 'Win Rate (%) Smoothed'] + 1.96 * df.loc[mask, 'Win Rate (%) Std'] / np.sqrt(n)
                df.loc[mask, 'CI Lower'] = df.loc[mask, 'Win Rate (%) Smoothed'] - 1.96 * df.loc[mask, 'Win Rate (%) Std'] / np.sqrt(n)
                df.loc[mask, 'CI Lower'] = df.loc[mask, 'CI Lower'].clip(lower=0)  # 負の値を0に置き換え
    
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
    if df.empty:
        print("エラー: プロットするデータがありません")
        return
    
    # データの移動平均を適用
    smoothed_df = apply_smoothing(df, window_size=30)
    
    # 信頼区間を計算
    smoothed_df = calculate_confidence_intervals(smoothed_df)
    
    # オプション: 初期エピソードをスキップ
    if skip_initial_episodes > 0:
        # 各アルゴリズムの最小エピソード番号を確認
        min_episodes = {}
        for algo in smoothed_df['Algorithm'].unique():
            min_ep = smoothed_df[smoothed_df['Algorithm'] == algo]['Episode'].min()
            min_episodes[algo] = min_ep + skip_initial_episodes
        
        # 初期エピソードをフィルタリング
        filtered_df = pd.DataFrame()
        for algo in smoothed_df['Algorithm'].unique():
            algo_data = smoothed_df[smoothed_df['Algorithm'] == algo]
            filtered_data = algo_data[algo_data['Episode'] >= min_episodes[algo]]
            filtered_df = pd.concat([filtered_df, filtered_data])
        
        # フィルタリングされたデータがある場合のみ置き換え
        if not filtered_df.empty:
            smoothed_df = filtered_df
            print(f"最初の {skip_initial_episodes} エピソードをスキップしました")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 利用可能なアルゴリズムを取得
    algorithms = smoothed_df['Algorithm'].unique()
    
    # カスタム色とスタイルの設定
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
    
    # マーカーを間引く間隔
    marker_step = max(1, len(smoothed_df) // 20)
    
    # アルゴリズムごとのプロット
    for algo in algorithms:
        # 表示用の短い名前（接尾辞Agentを除去）
        display_name = algo.replace('Agent', '')
        
        # アルゴリズムデータを取得
        algo_data = smoothed_df[smoothed_df['Algorithm'] == algo]
        
        # アルゴリズム用の色を取得（デフォルトを使用）
        color = colors.get(algo, sns.color_palette("husl", len(algorithms))[list(algorithms).index(algo)])
        marker = markers.get(algo, 'o')
        linestyle = linestyles.get(algo, '-')
        
        # 信頼区間プロット
        if 'CI Upper' in algo_data.columns and 'CI Lower' in algo_data.columns:
            plt.fill_between(
                algo_data['Episode'], 
                algo_data['CI Lower'], 
                algo_data['CI Upper'], 
                alpha=0.2, 
                color=color
            )
        
        # 平滑化されたラインプロット（すべてのポイント）
        plt.plot(
            algo_data['Episode'], 
            algo_data['Win Rate (%) Smoothed'], 
            label=display_name, 
            color=color, 
            linestyle=linestyle, 
            linewidth=2
        )
        
        # 間引いたマーカープロット
        plt.plot(
            algo_data.iloc[::marker_step]['Episode'], 
            algo_data.iloc[::marker_step]['Win Rate (%) Smoothed'], 
            color=color, 
            marker=marker, 
            linestyle='none', 
            markersize=8
        )
    
    # 最終値のアノテーション（「収束」ではなく「最終性能」として表記）
    final_annotations = []
    for algo in algorithms:
        algo_data = smoothed_df[smoothed_df['Algorithm'] == algo].sort_values('Episode')
        if not algo_data.empty:
            # 最終エピソードのデータポイント
            final_idx = algo_data['Episode'].idxmax()
            final_point = algo_data.loc[final_idx]
            final_ep = final_point['Episode']
            final_rate = final_point['Win Rate (%) Smoothed']
            
            # アノテーションテキスト
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
    
    # 特徴的なポイントを識別
    feature_annotations = []
    for algo in algorithms:
        algo_data = smoothed_df[smoothed_df['Algorithm'] == algo].sort_values('Episode')
        
        # 十分なデータポイントがある場合
        if len(algo_data) >= 20:
            # 勾配を計算（変化率）
            algo_data['slope'] = algo_data['Win Rate (%) Smoothed'].diff() / algo_data['Episode'].diff().clip(lower=1)
            
            # EAの場合、プラトー領域を検出
            if 'EA' in algo:
                # 傾きが非常に小さい領域を探す
                plateaus = algo_data[abs(algo_data['slope']) < 0.0005]
                if not plateaus.empty and len(plateaus) > 5:
                    # プラトーの中心点を取得
                    middle_idx = plateaus.iloc[len(plateaus) // 2].name
                    plateau_point = algo_data.loc[middle_idx]
                    feature_annotations.append({
                        'text': "EA Plateau",
                        'xy': (plateau_point['Episode'], plateau_point['Win Rate (%) Smoothed']),
                        'xytext': (plateau_point['Episode'] - 150, plateau_point['Win Rate (%) Smoothed'] + 0.60)
                    })
            
            # MARLの場合、性能低下領域を検出
            if 'MARL' in algo:
                # 後半のデータで負の勾配を探す
                second_half = algo_data.iloc[len(algo_data) // 2:]
                negative_slopes = second_half[second_half['slope'] < -0.0002]
                if not negative_slopes.empty and len(negative_slopes) > 3:
                    # 最も顕著な下降点
                    decline_idx = negative_slopes['slope'].idxmin()
                    decline_point = algo_data.loc[decline_idx]
                    feature_annotations.append({
                        'text': "MARL\nPerformance\nDegradation",
                        'xy': (decline_point['Episode'], decline_point['Win Rate (%) Smoothed']),
                        'xytext': (decline_point['Episode'] - 200, decline_point['Win Rate (%) Smoothed'] + 0)
                    })
    
    # アノテーションを追加
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
            fontsize=14
        )
    
    # グラフ設定
    plt.title('Learning Curves: Win Rate Comparison', fontsize=16)
    plt.xlabel('Training Episodes', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    
    # Y軸の範囲設定（引数に基づく）
    plt.ylim(0, max_y_limit)
    
    # X軸の範囲を確認して設定（グラフがきれいに表示されるように）
    min_x = smoothed_df['Episode'].min()
    max_x = smoothed_df['Episode'].max()
    plt.xlim(min_x, max_x)
    
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 初期エピソードをスキップした場合は、その説明を追加
    if skip_initial_episodes > 0:
        plt.figtext(
            0.5, 0.01,
            f"Note: Initial {skip_initial_episodes} episodes omitted. High win rates in early episodes reflect statistical variance due to small sample sizes.",
            ha="center", fontsize=9, style='italic'
        )
    
    # 出力ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, 'learning_curves')
    
    # 保存（高解像度で）
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')  # PDFも保存
    
    print(f"学習曲線を {output_base}.png と {output_base}.pdf に保存しました")
    
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
    else:
        print("エラー: データが読み込めないためプロットできません")

if __name__ == "__main__":
    main()