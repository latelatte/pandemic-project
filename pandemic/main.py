#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from datetime import datetime

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
    
    parser.add_argument('--episodes', type=int, default=10,
                       help='実行するエピソード数 (デフォルト: 10)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='ログと結果を保存するディレクトリ (デフォルト: ./logs)')
    parser.add_argument('--agents', nargs='+', 
                       default=['random', 'mcts', 'ea', 'marl'],
                       choices=['random', 'mcts', 'ea', 'marl'],
                       help='比較するエージェント (デフォルト: すべて)')
    parser.add_argument('--seed', type=int, default=None,
                       help='乱数シードを固定 (再現性のため)')
    
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

def main():
    """メイン実行関数"""
    args = parse_args()
    
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
    
    # エージェント戦略の準備
    strategies = get_agent_strategies(args.agents)
    if not strategies:
        print("エラー: 有効なエージェントが指定されていません")
        return
    
    print(f"選択されたエージェント: {[name for _, name in strategies]}")
    
    # 実験実行
    print(f"パンデミックシミュレーション実験を開始 ({args.episodes}エピソード)...")
    start_time = time.time()
    
    runner = SimulationRunner(n_episodes=args.episodes, log_dir=log_dir)
    runner.run_experiments(strategies)
    
    end_time = time.time()
    print(f"実験完了。所要時間: {end_time - start_time:.2f}秒")
    
    # 実験設定を保存
    import json
    with open(os.path.join(log_dir, "experiment_config.json"), "w") as f:
        json.dump({
            "episodes": args.episodes,
            "agents": [name for _, name in strategies],
            "seed": args.seed,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

if __name__ == "__main__":
    main()