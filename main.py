#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from datetime import datetime
import json
import sys

import tensorboard


from pandemic.simulation.pandemic import PandemicSimulation
from pandemic.simulation.runner import SimulationRunner
from pandemic.agents.baseline_agents import random_agent_strategy
from pandemic.agents.mcts_agent import mcts_agent_strategy
from pandemic.agents.ea_agent import ea_agent_strategy
from pandemic.agents.marl_agent import marl_agent_strategy

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='パンデミックシミュレーションでAIエージェントを比較')
    
    parser.add_argument('--episodes', type=int, default=30,
                       help='実行するエピソード数 (デフォルト: 10)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='ログと結果を保存するディレクトリ (デフォルト: ./logs)')
    parser.add_argument('--agents', nargs='+', 
                       default=['random', 'mcts', 'ea', 'marl'],
                       choices=['random', 'mcts', 'ea', 'marl'],
                       help='比較するエージェント (デフォルト: すべて)')
    parser.add_argument('--seed', type=int, default=None,
                       help='乱数シードを固定 (再現性のため)')
    parser.add_argument('--visualize', action='store_true',
                       help='実験結果を可視化', default=True)
    parser.add_argument('--players', type=int, default=4,
                       help='プレイヤー数 (デフォルト: 4)')
    parser.add_argument('--difficulty', type=str, default='normal',
                       choices=['easy', 'normal', 'hard'],
                       help='ゲーム難易度 (デフォルト: normal)')
    
    return parser.parse_args()

def get_agent_strategies(agent_names):
    """指定されたエージェント名から戦略関数のリストを作成"""
    agent_map = {
        'random': (random_agent_strategy, "RandomAgent"),
        'mcts': (mcts_agent_strategy, "MCTSAgent"),
        'ea': (ea_agent_strategy, "EAAgent"),
        'marl': (marl_agent_strategy, "MARLAgent")
    }
    
    strategies = []
    for name in agent_names:
        if name in agent_map:
            strategies.append(agent_map[name])
        else:
            print(f"警告: 未知のエージェント '{name}' はスキップされました")
    
    return strategies

def setup_experiment_dir(base_log_dir):
    """タイムスタンプ付きの実験ディレクトリを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_log_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def check_config_files():
    # パッケージのルートディレクトリを特定
    config_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pandemic", "config"
    )
    
    print(f"Checking config files in: {config_dir}")
    
    config_files = [
        "cities_config.json",
        "diseases_config.json",
        "roles_config.json",
        "game_config.json"
    ]
    
    all_exist = True
    for filename in config_files:
        filepath = os.path.join(config_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} exists")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"  JSON is valid, keys: {list(data.keys())}")
            except json.JSONDecodeError:
                print(f"✗ {filename} contains invalid JSON")
                all_exist = False
        else:
            print(f"✗ {filename} does not exist")
            all_exist = False
    
    return all_exist

def main():
    """メイン実行関数"""
    args = parse_args()
    
    # 設定ディレクトリの絶対パス
    config_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pandemic", "config"
    )
    print(f"設定ディレクトリ: {config_dir}")
    
    # 乱数シードの設定（指定されていれば）
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"乱数シードを {args.seed} に設定")
    
    # 実験ディレクトリのセットアップ
    log_dir = setup_experiment_dir(args.log_dir)
    print(f"実験結果は {log_dir} に保存されます")
    
    # TensorBoardの初期化
    tb_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    
    # エージェント戦略の準備
    strategies = get_agent_strategies(args.agents)
    if not strategies:
        print("エラー: 有効なエージェントが指定されていません")
        return
    
    print(f"選択されたエージェント: {[name for _, name in strategies]}")
    
    # 実験実行
    print(f"パンデミックシミュレーション実験を開始 ({args.episodes}エピソード)...")
    start_time = time.time()
    
    # 設定ディレクトリを明示的に渡す
    runner = SimulationRunner(
        n_episodes=args.episodes, 
        log_dir=log_dir,
        num_players=args.players,
        difficulty=args.difficulty
    )
    results = runner.run_experiments(strategies, config_dir=config_dir)
    
    end_time = time.time()
    print(f"実験完了。所要時間: {end_time - start_time:.2f}秒")
    
    # 実験設定を保存
    with open(os.path.join(log_dir, "experiment_config.json"), "w") as f:
        json.dump({
            "episodes": args.episodes,
            "agents": [name for _, name in strategies],
            "seed": args.seed,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    # 実験終了後、可視化（必要であれば）
    if args.visualize:
        print("結果を可視化しています...")
        from visualization.performance_charts import create_performance_charts
        create_performance_charts(log_dir, os.path.join(log_dir, "plots"))
        print(f"可視化結果を {log_dir}/plots に保存しました")

if __name__ == "__main__":
    if check_config_files():
        print("All config files exist and are valid.")
        main()
    else:
        print("Some config files are missing or invalid.")
        sys.exit(1)