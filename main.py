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
    """for command line arguments"""
    parser = argparse.ArgumentParser(description='pandemic simulation')
    
    parser.add_argument('-e', '--episodes', type=int, default=20,
                       help='number of episodes to run (default: 100)')
    parser.add_argument('-l', '--log-dir', type=str, default='./logs',
                       help='directory to save logs (default: ./logs)')
    parser.add_argument('-a', '--agents', nargs='+', 
                       default=['random', 'mcts', 'ea', 'marl'],
                       choices=['random', 'mcts', 'ea', 'marl'],
                       help='list of agent strategies to use (default: all)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                       help='random seed for reproducibility (default: None)')
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='make plots', default=True)
    parser.add_argument('-p', '--players', type=int, default=4,
                       help='number of players (default: 4)')
    parser.add_argument('-d', '--difficulty', type=str, default='easy',
                       choices=['easy', 'normal', 'hard'],
                       help='game difficulty (default: normal)')
    
    return parser.parse_args()

def get_agent_strategies(agent_names):
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
            print(f"WARN: unknown agent '{name}' skipped.")
    
    return strategies

def get_agent_names(agent_strategies):
    names = [name.replace('Agent', '') for _, name in agent_strategies]
    if len(names) == 1:
        return names[0]
    else:
        return "-".join(names)
    
def setup_experiment_dir(base_log_dir, agent_strategies):
    """make experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = get_agent_names(agent_strategies)
    exp_dir = os.path.join(base_log_dir, f"experiment_{agent_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def check_config_files():
    config_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pandemic", "config"
    )
    
    # print(f"Checking config files in: {config_dir}")
    
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
            # print(f"✓ {filename} exists")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                # print(f"  JSON is valid, keys: {list(data.keys())}")
            except json.JSONDecodeError:
                print(f"✗ {filename} contains invalid JSON")
                all_exist = False
        else:
            print(f"✗ {filename} does not exist")
            all_exist = False
    
    return all_exist

def main():
    """main function"""
    args = parse_args()
    
    config_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pandemic", "config"
    )
    # print(f"config directory: {config_dir}")
    
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"set {args.seed} as random seed")
    
    strategies = get_agent_strategies(args.agents)
    log_dir = setup_experiment_dir(args.log_dir, strategies)
    print(f"experiments logs will save to {log_dir} ")
    
    tb_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    
    if not strategies:
        print("error: no valid agent strategies selected")
        return
    
    print(f"selected agent: {[name for _, name in strategies]}")

    print(f"starting episode {args.episodes}")
    start_time = time.time()
    
    runner = SimulationRunner(
        n_episodes=args.episodes, 
        log_dir=log_dir,
        num_players=args.players,
        difficulty=args.difficulty
    )
    results = runner.run_experiments(strategies, config_dir=config_dir)
    
    end_time = time.time()
    print(f"Finished. took: {end_time - start_time:.2f} seconds")
    
    with open(os.path.join(log_dir, "experiment_config.json"), "w") as f:
        json.dump({
            "episodes": args.episodes,
            "agents": [name for _, name in strategies],
            "seed": args.seed,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    if args.visualize:
        #print("visualizing results...")
        from visualization.performance_charts import create_performance_charts
        from visualization.learning_curves import create_learning_curves
        
        create_performance_charts(log_dir, os.path.join(log_dir, "plots"))
        create_learning_curves(log_dir, os.path.join(log_dir, "plots"))
        #print(f"images saved to {log_dir}/plots")

if __name__ == "__main__":
    if check_config_files():
        print("All config files exist and are valid.")
        main()
    else:
        print("Some config files are missing or invalid.")
        sys.exit(1)