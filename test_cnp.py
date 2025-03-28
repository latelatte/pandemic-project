#!/usr/bin/env python3
# test_cnp_analysis.py - 複数ログファイルを使用した簡易テスト

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from visualization.cnp_analysis import create_cnp_visualization

def find_experiment_logs(base_dir="./logs"):
    """利用可能な実験ログを探す"""
    experiments = []
    
    # 実験ディレクトリを探す
    for exp_dir in glob.glob(os.path.join(base_dir, "experiment_*")):
        metrics_file = os.path.join(exp_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                try:
                    data = json.load(f)
                    # どのエージェントのログか判断するための情報を抽出
                    agents = list(data.get("agent_performance", {}).keys())
                    primary_agent = agents[0] if agents else "Unknown"
                    experiments.append((exp_dir, metrics_file, primary_agent))
                except:
                    print(f"警告: {metrics_file}は無効なJSONファイルです")
    
    return experiments

def create_aggregated_data(selected_experiments):
    """複数の実験データを集約"""
    all_agent_data = {}
    
    for exp_dir, metrics_file, primary_agent in selected_experiments:
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # エージェント情報を抽出
            for agent, metrics in data.get("agent_performance", {}).items():
                if agent not in all_agent_data:
                    all_agent_data[agent] = {
                        "win_rate": data.get("win_rate", 0.5) * 100,
                        "avg_time_ms": metrics.get("avg_time_ms", 100),
                        "memory_mb": data.get("resource_usage", {}).get(agent, {}).get("avg_memory_mb", 50),
                        "episodes": data.get("episodes", 1000),
                        "source_dir": os.path.basename(exp_dir)
                    }
        except Exception as e:
            print(f"エラー: {metrics_file}の処理中にエラーが発生しました: {e}")
    
    return all_agent_data

def simulate_dual_evaluation_data(agent_data):
    """集約されたエージェントデータからデュアル評価データを模倣"""
    mock_data = {
        "fixed_episodes": {},
        "fixed_resource": {},
        "comparison": {},
        "cnp_metrics": {},
        "settings": {"fixed_episodes": 1000}
    }
    
    for agent, data in agent_data.items():
        win_rate = data["win_rate"]
        avg_time = data["avg_time_ms"]
        memory_mb = data["memory_mb"]
        
        # 固定エピソード評価のデータ
        mock_data["fixed_episodes"][agent] = {
            "win_rate": win_rate,
            "avg_time_ms": avg_time,
            "memory_mb": memory_mb,
            "episodes": data["episodes"]
        }
        
        # 固定リソース評価のデータを模倣
        # 各エージェントの特性に基づいて調整
        if "MCTS" in agent:
            # MCTSは通常より早く収束するが、より多くのメモリを使用
            resource_ratio = 1.2
            time_ratio = 0.8
        elif "EA" in agent:
            # EAは進化するため、より多くのエピソードが可能
            resource_ratio = 0.7
            time_ratio = 0.9
        elif "MARL" in agent:
            # MARLは学習に時間がかかり、メモリも多く使用
            resource_ratio = 1.3
            time_ratio = 1.4
        else:
            # その他のエージェント（ランダムなど）
            resource_ratio = 1.0
            time_ratio = 1.0
        
        adj_episodes = int(data["episodes"] * (1 / time_ratio))
        adj_win_rate = win_rate * resource_ratio * 0.9  # リソース固定では勝率が若干低下
        
        mock_data["fixed_resource"][agent] = {
            "win_rate": adj_win_rate,
            "avg_time_ms": avg_time * time_ratio,
            "memory_mb": memory_mb * resource_ratio,
            "episodes": adj_episodes
        }
        
        # CNPメトリクスを計算
        memory_gb = memory_mb / 1024
        time_hrs = avg_time / (3600 * 1000)
        resource_memory_gb = memory_mb * resource_ratio / 1024
        resource_time_hrs = avg_time * time_ratio / (3600 * 1000)
        
        fixed_episodes_cnp = win_rate / (memory_gb * time_hrs)
        fixed_resource_cnp = adj_win_rate / (resource_memory_gb * resource_time_hrs)
        
        mock_data["cnp_metrics"][agent] = {
            "fixed_episodes_cnp": fixed_episodes_cnp,
            "fixed_resource_cnp": fixed_resource_cnp
        }
    
    return mock_data

def main():
    # 利用可能な実験ログを探す
    experiments = find_experiment_logs()
    
    if not experiments:
        print("警告: 実験ログが見つかりません。")
        return
    
    print(f"{len(experiments)}件の実験ログが見つかりました:")
    for i, (exp_dir, metrics_file, primary_agent) in enumerate(experiments):
        print(f"{i+1}. {os.path.basename(exp_dir)} - 主要エージェント: {primary_agent}")
    
    # 複数選択の例（すべてのエージェント用ログを選択）
    # 実際の使用では、ユーザー入力でインデックス選択も可能
    selected_indices = list(range(len(experiments)))
    selected_experiments = [experiments[i] for i in selected_indices]
    
    print(f"\n{len(selected_experiments)}件の実験を分析対象として選択しました")
    
    # 複数の実験から集約データを作成
    agent_data = create_aggregated_data(selected_experiments)
    
    if not agent_data:
        print("エラー: エージェントデータの作成に失敗しました。")
        return
    
    print(f"集約されたエージェント: {', '.join(agent_data.keys())}")
    
    # 模擬デュアル評価データを作成
    mock_data = simulate_dual_evaluation_data(agent_data)
    
    # 出力ディレクトリを作成
    output_dir = os.path.join("./logs", "cnp_analysis_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # 可視化を実行
    print("\nCNP可視化を作成中...")
    create_cnp_visualization(mock_data, output_dir)
    print(f"可視化が完了しました。結果は {output_dir} に保存されました。")
    
    # エージェントデータの要約を表示
    print("\n集約されたエージェントデータ:")
    for agent, data in agent_data.items():
        print(f"- {agent}: 勝率={data['win_rate']:.1f}%, 平均時間={data['avg_time_ms']:.1f}ms, メモリ={data['memory_mb']:.1f}MB")

if __name__ == "__main__":
    main()