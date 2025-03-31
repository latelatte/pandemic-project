"""
Integrated Evaluation Framework: Combines convergence training with dual evaluation

This module implements a three-stage evaluation process:
1. Convergence Training: Train agents until convergence or time limit
2. Fixed Episodes Evaluation: Test converged agents with fixed number of episodes
3. Fixed Resource Evaluation: Test converged agents with fixed computational resources

This provides a comprehensive assessment of agent performance and efficiency.
"""

import time
import os
import datetime
import json
import psutil
import threading
import math
import numpy as n
import shutil
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pandemic.simulation.pandemic import PandemicSimulation
from pandemic.simulation.runner import SimulationRunner
from pandemic.utils.metrics_utils import MetricsCollector
from pandemic.utils.resource_utils import ResourceMonitor


from pandemic.agents.baseline_agents import random_agent_strategy
from pandemic.agents.mcts_agent import mcts_agent_strategy
from pandemic.agents.ea_agent import ea_agent_strategy
from pandemic.agents.marl_agent import marl_agent_strategy


class ConvergenceDetector:
    """
    Detects when an agent's performance has converged based on win rate stability.
    
    Attributes:
        window_size: Number of episodes for win rate calculation
        threshold: Maximum win rate change to consider stable
        stability_periods: Number of consecutive stable periods needed for convergence
        win_history: Record of win/loss results
        stable_periods: Counter for consecutive stable periods
        converged: Whether convergence has been detected
        convergence_time: Time taken to reach convergence
        convergence_episode: Episode at which convergence was detected
    """
    def __init__(self, window_size=100, threshold=0.02, stability_periods=3):
        self.window_size = window_size
        self.threshold = threshold
        self.stability_periods = stability_periods
        self.win_history = []
        self.stable_periods = 0
        self.converged = False
        self.convergence_time = None
        self.convergence_episode = None
        self.start_time = time.time()
    
    def update(self, win, episode):
        """
        Update with new win/loss result and check for convergence
        
        Args:
            win: Boolean indicating win (True) or loss (False)
            episode: Current episode number
            
        Returns:
            bool: True if convergence newly detected, False otherwise
        """
        self.win_history.append(1 if win else 0)
        
        # Need at least 2 windows of data to compare
        if len(self.win_history) >= self.window_size * 2:
            recent = self.win_history[-self.window_size:]
            previous = self.win_history[-(self.window_size*2):-self.window_size]
            
            recent_win_rate = sum(recent) / self.window_size
            previous_win_rate = sum(previous) / self.window_size
            
            change = abs(recent_win_rate - previous_win_rate)
            
            if change < self.threshold:
                self.stable_periods += 1
                if self.stable_periods >= self.stability_periods and not self.converged:
                    self.converged = True
                    self.convergence_time = time.time() - self.start_time
                    self.convergence_episode = episode
                    return True
            else:
                self.stable_periods = 0
        
        return False
    
    def get_current_win_rate(self):
        """
        Calculate current win rate based on recent episodes
        
        Returns:
            float: Current win rate (0.0-1.0)
        """
        if not self.win_history:
            return 0.0
            
        window = min(self.window_size, len(self.win_history))
        return sum(self.win_history[-window:]) / window


class ConvergenceSimulationRunner(SimulationRunner):
    """
    Extended simulation runner that detects convergence
    
    This class extends the standard SimulationRunner with the ability
    to detect when agent performance has converged, allowing for
    early termination of training.
    
    Attributes:
        convergence_detector: Detector for convergence
        max_time: Maximum time limit in seconds
        start_time: Time when simulation started
    """
    
    def __init__(self, n_episodes=100, log_dir="./logs", num_players=4, 
                 difficulty="normal", convergence_detector=None, max_time=86400):
        super().__init__(n_episodes, log_dir, num_players, difficulty)
        self.convergence_detector = convergence_detector
        self.max_time = max_time
        self.start_time = time.time()
    
    def run_experiments(self, strategies, config_dir=None):
        """
        Run simulation experiments with convergence detection
        
        Args:
            strategies: List of strategy tuples (func, name)
            config_dir: Directory for configuration files
            
        Returns:
            dict: Metrics data including convergence information
        """
        agent_names = [name for _, name in strategies]
        self.metrics = MetricsCollector(agent_names)
        
        for ep in range(self.n_episodes):
            # Check time limit
            if time.time() - self.start_time > self.max_time:
                print(f"Maximum time limit reached after {ep} episodes")
                break
                
            sim = PandemicSimulation(*strategies, 
                                 config_dir=config_dir, 
                                 num_players=self.num_players,
                                 difficulty=self.difficulty)
            
            # Use timed strategies for performance measurement
            for p in sim.players:
                original_func = p.strategy_func
                strategy_name = p.strategy_name
                p.strategy_func = self.make_timed_strategy(original_func, strategy_name)
            
            sim.run_game()
            
            win = sim.is_win_condition()
            if win:
                print(f"Episode {ep+1}: WIN.")
                self.wins += 1
            else:
                print(f"Episode {ep+1}: LOSE.")
                self.losses += 1
            
            self.metrics.record_game_metrics(sim, win)
            
            # Update convergence detector if available
            if self.convergence_detector and self.convergence_detector.update(win, ep):
                print(f"Convergence detected at episode {ep+1}!")
                # Optionally terminate early
                break
            
            # Periodically evaluate progress
            if ep % 100 == 0 and ep > 0:
                self.evaluate_agent_progress(ep)
        
        # Prepare metrics data
        metrics_summary = self.metrics.get_summary()
        resource_summary = self.resource_monitor.get_summary()
        
        metrics_data = {
            "win_rate": self.wins / max(1, (self.wins + self.losses)),
            "total_episodes": ep + 1,  # Actual number of episodes run
            "avg_turns": metrics_summary["avg_turns"],
            "avg_outbreaks": metrics_summary["avg_outbreaks"],
            "agent_performance": metrics_summary["agent_performance"],
            "resource_usage": resource_summary
        }
        
        return metrics_data


class IntegratedEvaluationFramework:
    """
    Integrated evaluation framework with convergence detection
    
    This framework provides a comprehensive evaluation approach:
    1. Train agents until convergence
    2. Evaluate converged agents with fixed episodes
    3. Evaluate converged agents with fixed resources
    
    This allows for fair comparison of both training efficiency
    and execution performance.
    
    Attributes:
        log_dir: Directory for saving evaluation results
        config_dir: Directory for configuration files
        resource_monitor: Monitor for resource usage
        settings: Evaluation settings
    """
    
    def __init__(self, log_dir="./evaluations", config_dir=None):
        """
        Initialize the evaluation framework
        
        Args:
            log_dir: Directory to save evaluation results
            config_dir: Directory for configuration files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if config_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_dir = os.path.join(script_dir, "pandemic", "config")
        else:
            self.config_dir = config_dir
            
        print(f"Using config directory: {self.config_dir}")
        
        self.resource_monitor = ResourceMonitor()
        
        # Timing metrics
        self.start_time = None
        self.end_time = None
        
        # Results storage
        self.convergence_results = {}
        self.fixed_episodes_results = {}
        self.fixed_resource_results = {}
        
        # Default settings
        self.default_episodes = 5000
        self.default_resource_time = 24 * 60 * 60  # 24 hours in seconds
        self.eval_episodes = 1000
        
        # Load settings file
        self.settings = self._load_evaluation_config()
        
    def _load_evaluation_config(self):
        """
        Load evaluation settings or use defaults
        
        Returns:
            dict: Evaluation settings
        """
        settings_file = os.path.join(self.config_dir, "evaluation_config.json")
        default_settings = {
            "fixed_episodes": self.default_episodes,
            "fixed_resource_time": self.default_resource_time,
            "eval_episodes": self.eval_episodes,
            "convergence_max_episodes": 50000,
            "convergence_max_time": 72 * 60 * 60,  # 72 hours
            "convergence_window": 100,
            "convergence_threshold": 0.02,
            "convergence_stability": 3,
            "num_players": 4,
            "difficulty": "normal",
            "cross_validation_folds": 5,
            "seed": 42,
            "parallel_evaluation": True,
            "max_threads": 4
        }
        
        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                print(f"Loaded evaluation settings from {settings_file}")
                
                # Fill missing values with defaults
                for key, value in default_settings.items():
                    if key not in settings:
                        settings[key] = value
                        
                return settings
            else:
                print(f"Settings file {settings_file} not found. Using defaults.")
                return default_settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            return default_settings
    
    def run_convergence_evaluation(self, agent_strategies):
        """
        Run evaluation until convergence is detected or limits are reached
        
        Args:
            agent_strategies: List of agent strategy tuples [(func, name), ...]
            
        Returns:
            dict: Evaluation results with convergence metrics
        """
        max_episodes = self.settings["convergence_max_episodes"]
        max_time = self.settings["convergence_max_time"]
        window_size = self.settings["convergence_window"]
        threshold = self.settings["convergence_threshold"]
        stability_periods = self.settings["convergence_stability"]
        
        print(f"\n===== Starting Convergence Evaluation =====")
        print(f"Max episodes: {max_episodes}, Max time: {max_time/3600:.1f} hours")
        print(f"Convergence parameters: window={window_size}, threshold={threshold}, stability={stability_periods}")
        
        self.start_time = time.time()
        
        # Create evaluation directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(self.log_dir, f"convergence_eval_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)
        
        results = {}
        
        for strategy_func, strategy_name in agent_strategies:
            print(f"\nEvaluating convergence for: {strategy_name}")
            strategy_dir = os.path.join(eval_dir, strategy_name)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Configure convergence detector
            convergence_detector = ConvergenceDetector(
                window_size=window_size,
                threshold=threshold,
                stability_periods=stability_periods
            )
            
            # Run with convergence detection
            runner = ConvergenceSimulationRunner(
                n_episodes=max_episodes,
                log_dir=strategy_dir,
                num_players=self.settings["num_players"],
                difficulty=self.settings["difficulty"],
                convergence_detector=convergence_detector,
                max_time=max_time
            )
            
            metrics = runner.run_experiments([(strategy_func, strategy_name)], config_dir=self.config_dir)
            
            # Add convergence information
            metrics["converged"] = convergence_detector.converged
            metrics["convergence_episode"] = convergence_detector.convergence_episode
            metrics["convergence_time"] = convergence_detector.convergence_time
            metrics["final_win_rate"] = convergence_detector.get_current_win_rate()
            
            # Store results
            results[strategy_name] = metrics
            
            # Display results
            if metrics["converged"]:
                print(f"  Convergence detected at episode {metrics['convergence_episode']}")
                print(f"  Convergence time: {metrics['convergence_time']:.2f} seconds")
                print(f"  Final win rate: {metrics['final_win_rate'] * 100:.2f}%")
            else:
                print(f"  No convergence detected within {max_episodes} episodes or {max_time/3600:.1f} hours")
                print(f"  Current win rate: {metrics['final_win_rate'] * 100:.2f}%")
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print(f"\nConvergence Evaluation completed in {total_time/3600:.2f} hours")
        
        # Save overall results
        summary = {
            "evaluation_type": "convergence",
            "max_episodes": max_episodes,
            "max_time": max_time,
            "total_time": total_time,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(os.path.join(eval_dir, "convergence_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.convergence_results = results
        return results
    
    def run_fixed_episodes_evaluation(self, agent_strategies, episodes=None, use_converged=False, converged_dir=None):
        """
        Run evaluation with a fixed number of episodes
        
        Args:
            agent_strategies: List of agent strategy tuples [(func, name), ...]
            episodes: Number of episodes to run
            use_converged: Whether to use converged agents
            converged_dir: Directory with converged agent states
            
        Returns:
            dict: Evaluation results
        """
        if episodes is None:
            episodes = self.settings["fixed_episodes"]
            
        print(f"\n===== Starting Fixed Episodes Evaluation: {episodes} episodes =====")
        self.start_time = time.time()
        
        # Create evaluation directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(self.log_dir, f"fixed_episodes_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)
        
        results = {}
        
        for strategy_func, strategy_name in agent_strategies:
            print(f"\nEvaluating: {strategy_name}")
            strategy_dir = os.path.join(eval_dir, strategy_name)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # If using converged agents, copy state files
            if use_converged and converged_dir:
                agent_file = f"{strategy_name.lower()}_agent_state.{'pt' if 'MARL' in strategy_name else 'pkl'}"
                source_path = os.path.join(converged_dir, agent_file)
                target_path = os.path.join("./agents_state", agent_file)
                
                if os.path.exists(source_path):
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy(source_path, target_path)
                    print(f"  Using converged agent state from {source_path}")
                else:
                    print(f"  Warning: Converged agent state not found at {source_path}")
            
            # Run agent evaluation
            runner = SimulationRunner(
                n_episodes=episodes,
                log_dir=strategy_dir,
                num_players=self.settings["num_players"],
                difficulty=self.settings["difficulty"]
            )
            metrics = runner.run_experiments([(strategy_func, strategy_name)], config_dir=self.config_dir)
            
            # Add resource usage information
            if "resource_usage" not in metrics:
                metrics["resource_usage"] = {}
            metrics["resource_usage"][strategy_name] = self.resource_monitor.get_agent_summary(strategy_name)
            
            # Store results
            results[strategy_name] = metrics
            
            # Display results
            win_rate = metrics.get("win_rate", 0) * 100
            avg_time = metrics.get("agent_performance", {}).get(strategy_name, {}).get("avg_time_ms", 0)
            memory_mb = metrics.get("resource_usage", {}).get(strategy_name, {}).get("avg_memory_mb", 0)
            
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Avg Decision Time: {avg_time:.2f} ms")
            print(f"  Avg Memory Usage: {memory_mb:.2f} MB")
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print(f"\nFixed Episodes Evaluation completed in {total_time/3600:.2f} hours")
        
        # Save overall results
        summary = {
            "evaluation_type": "fixed_episodes",
            "episodes": episodes,
            "use_converged": use_converged,
            "total_time": total_time,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(os.path.join(eval_dir, "evaluation_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.fixed_episodes_results = results
        return results
    
    def run_fixed_resource_evaluation(self, agent_strategies, resource_time=None, use_converged=False, converged_dir=None):
        """
        Run evaluation with fixed computational resources (time-based)
        
        Args:
            agent_strategies: List of agent strategy tuples [(func, name), ...]
            resource_time: Time limit in seconds
            use_converged: Whether to use converged agents
            converged_dir: Directory with converged agent states
            
        Returns:
            dict: Evaluation results
        """
        if resource_time is None:
            resource_time = self.settings["fixed_resource_time"]
            
        print(f"\n===== Starting Fixed Resource Evaluation: {resource_time/3600:.2f} hours =====")
        self.start_time = time.time()
        
        # Create evaluation directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(self.log_dir, f"fixed_resource_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)
        
        results = {}
        
        # Run each agent with time limit
        for strategy_func, strategy_name in agent_strategies:
            print(f"\nEvaluating: {strategy_name}")
            strategy_dir = os.path.join(eval_dir, strategy_name)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # If using converged agents, copy state files
            if use_converged and converged_dir:
                agent_file = f"{strategy_name.lower()}_agent_state.{'pt' if 'MARL' in strategy_name else 'pkl'}"
                source_path = os.path.join(converged_dir, agent_file)
                target_path = os.path.join("./agents_state", agent_file)
                
                if os.path.exists(source_path):
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy(source_path, target_path)
                    print(f"  Using converged agent state from {source_path}")
                else:
                    print(f"  Warning: Converged agent state not found at {source_path}")
            
            # Set up the time-constrained evaluator
            agent_metrics = self._run_time_constrained_evaluation(
                strategy_func, 
                strategy_name, 
                strategy_dir, 
                resource_time
            )
            
            # Store results
            results[strategy_name] = agent_metrics
            
            # Display results
            win_rate = agent_metrics.get("win_rate", 0) * 100
            avg_time = agent_metrics.get("agent_performance", {}).get(strategy_name, {}).get("avg_time_ms", 0)
            memory_mb = agent_metrics.get("resource_usage", {}).get(strategy_name, {}).get("avg_memory_mb", 0)
            episodes_completed = agent_metrics.get("total_episodes", 0)
            
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Episodes Completed: {episodes_completed}")
            print(f"  Avg Decision Time: {avg_time:.2f} ms")
            print(f"  Avg Memory Usage: {memory_mb:.2f} MB")
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print(f"\nFixed Resource Evaluation completed in {total_time/3600:.2f} hours")
        
        # Save overall results
        summary = {
            "evaluation_type": "fixed_resource",
            "resource_time": resource_time,
            "use_converged": use_converged,
            "total_time": total_time,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(os.path.join(eval_dir, "evaluation_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.fixed_resource_results = results
        return results
    
    def _run_time_constrained_evaluation(self, strategy_func, strategy_name, log_dir, time_limit):
        """
        Run evaluation with a time constraint
        
        Args:
            strategy_func: Agent strategy function
            strategy_name: Agent name
            log_dir: Directory to save logs
            time_limit: Time limit in seconds
            
        Returns:
            dict: Evaluation metrics
        """
        # Start with a high episode count - will be terminated by time
        max_episodes = 100000
        
        # Create a stop flag for the runner
        stop_flag = threading.Event()
        
        # Create a metrics collector
        metrics_collector = MetricsCollector([strategy_name])
        
        # Thread to monitor time and set stop flag
        def time_monitor():
            start = time.time()
            while time.time() - start < time_limit:
                time.sleep(1)
            stop_flag.set()
            print(f"Time limit reached for {strategy_name}. Stopping evaluation.")
        
        # Start the time monitor thread
        monitor_thread = threading.Thread(target=time_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Create a modified runner that respects the stop flag
        class TimeLimitedRunner(SimulationRunner):
            def run_experiments(self, strategies, config_dir=None):
                self.metrics = metrics_collector
                self.episodes_completed = 0
                
                for ep in range(self.n_episodes):
                    if stop_flag.is_set():
                        print(f"Stopped after {ep} episodes due to time constraint")
                        break
                        
                    sim = PandemicSimulation(*strategies, 
                                          config_dir=config_dir, 
                                          num_players=self.num_players,
                                          difficulty=self.difficulty)
                    
                    # Run episode and collect metrics
                    sim.run_game()
                    win = sim.is_win_condition()
                    
                    self.metrics.record_game_metrics(sim, win)
                    
                    if win:
                        self.wins += 1
                    else:
                        self.losses += 1
                    
                    # Update episode count for reporting
                    self.episodes_completed = ep + 1
                
                metrics_summary = self.metrics.get_summary()
                resource_summary = self.resource_monitor.get_summary()
                
                metrics_data = {
                    "win_rate": self.wins / max(1, self.episodes_completed),
                    "total_episodes": self.episodes_completed,
                    "avg_turns": metrics_summary.get("avg_turns", 0),
                    "avg_outbreaks": metrics_summary.get("avg_outbreaks", 0),
                    "agent_performance": metrics_summary.get("agent_performance", {}),
                    "resource_usage": resource_summary
                }
                
                return metrics_data
        
        # Create and run the time-limited runner
        runner = TimeLimitedRunner(
            n_episodes=max_episodes,
            log_dir=log_dir,
            num_players=self.settings["num_players"],
            difficulty=self.settings["difficulty"]
        )
        
        metrics = runner.run_experiments([(strategy_func, strategy_name)], config_dir=self.config_dir)
        
        return metrics
    
    def run_integrated_evaluation(self, agent_strategies):
        """
        Integrated evaluation workflow:
        1. Train agents until convergence
        2. Save converged agents
        3. Evaluate converged agents with fixed episodes and resources
        
        Args:
            agent_strategies: List of agent strategy tuples [(func, name), ...]
            
        Returns:
            tuple: (convergence_results, fixed_episodes_results, fixed_resource_results)
        """
        print("\n===== STARTING INTEGRATED EVALUATION WORKFLOW =====")
        
        # Step 1: Convergence training
        convergence_results = self.run_convergence_evaluation(agent_strategies)
        
        # Step 2: Save converged agents
        converged_agents_dir = os.path.join(self.log_dir, "converged_agents")
        os.makedirs(converged_agents_dir, exist_ok=True)
        
        for agent_name, results in convergence_results.items():
            # Copy agent state files to converged_agents directory
            agent_file = f"{agent_name.lower()}_agent_state.{'pt' if 'MARL' in agent_name else 'pkl'}"
            source_path = os.path.join("./agents_state", agent_file)
            target_path = os.path.join(converged_agents_dir, agent_file)
            
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
                print(f"Saved converged agent state for {agent_name} to {target_path}")
            else:
                print(f"Warning: Could not find agent state at {source_path}")
        
        # Step 3: Evaluate converged agents with fixed episodes
        print("\n----- Evaluating Converged Agents with Fixed Episodes -----")
        fixed_episodes_results = self.run_fixed_episodes_evaluation(
            agent_strategies, 
            use_converged=True, 
            converged_dir=converged_agents_dir
        )
        
        # Step 4: Evaluate converged agents with fixed resources
        print("\n----- Evaluating Converged Agents with Fixed Resources -----")
        fixed_resource_results = self.run_fixed_resource_evaluation(
            agent_strategies,
            use_converged=True,
            converged_dir=converged_agents_dir
        )
        
        # Step 5: Generate comprehensive evaluation report
        self.generate_integrated_report(
            convergence_results, 
            fixed_episodes_results, 
            fixed_resource_results
        )
        
        return convergence_results, fixed_episodes_results, fixed_resource_results
    
    def calculate_cnp_metrics(self, fixed_episodes_results=None, fixed_resource_results=None, convergence_results=None):
        """
        Calculate CNP (Cost-Normalized Performance) metrics
        
        Args:
            fixed_episodes_results: Results from fixed episodes evaluation
            fixed_resource_results: Results from fixed resource evaluation
            convergence_results: Results from convergence evaluation
            
        Returns:
            dict: CNP metrics
        """
        if fixed_episodes_results is None:
            fixed_episodes_results = self.fixed_episodes_results
            
        if fixed_resource_results is None:
            fixed_resource_results = self.fixed_resource_results
            
        if convergence_results is None:
            convergence_results = self.convergence_results
            
        cnp_metrics = {}
        
        # Process fixed episodes results
        for agent_name, metrics in fixed_episodes_results.items():
            win_rate = metrics.get("win_rate", 0) * 100  # Convert to percentage
            avg_time_ms = metrics.get("agent_performance", {}).get(agent_name, {}).get("avg_time_ms", 0)
            memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_memory_mb", 0)
            
            # Avoid division by zero
            if memory_mb <= 0 or avg_time_ms <= 0:
                cnp = 0
            else:
                # Convert units for CNP calculation
                memory_gb = memory_mb / 1024.0
                time_hrs = avg_time_ms / (3600 * 1000)
                cnp = win_rate / (memory_gb * time_hrs)
            
            cnp_metrics[agent_name] = {
                "fixed_episodes_cnp": cnp,
                "fixed_episodes_win_rate": win_rate,
                "fixed_episodes_memory_gb": memory_mb / 1024.0,
                "fixed_episodes_time_hrs": avg_time_ms / (3600 * 1000)
            }
        
        # Process fixed resource results
        for agent_name, metrics in fixed_resource_results.items():
            if agent_name not in cnp_metrics:
                cnp_metrics[agent_name] = {}
                
            win_rate = metrics.get("win_rate", 0) * 100  # Convert to percentage
            avg_time_ms = metrics.get("agent_performance", {}).get(agent_name, {}).get("avg_time_ms", 0)
            memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_memory_mb", 0)
            episodes = metrics.get("total_episodes", 0)
            
            # Avoid division by zero
            if memory_mb <= 0 or avg_time_ms <= 0:
                cnp = 0
            else:
                # Convert units for CNP calculation
                memory_gb = memory_mb / 1024.0
                time_hrs = avg_time_ms / (3600 * 1000)
                cnp = win_rate / (memory_gb * time_hrs)
            
            cnp_metrics[agent_name].update({
                "fixed_resource_cnp": cnp,
                "fixed_resource_win_rate": win_rate,
                "fixed_resource_memory_gb": memory_mb / 1024.0,
                "fixed_resource_time_hrs": avg_time_ms / (3600 * 1000),
                "fixed_resource_episodes": episodes
            })
        
        # Process convergence results if available
        if convergence_results:
            for agent_name, metrics in convergence_results.items():
                if agent_name not in cnp_metrics:
                    cnp_metrics[agent_name] = {}
                
                converged = metrics.get("converged", False)
                convergence_episode = metrics.get("convergence_episode", None)
                convergence_time = metrics.get("convergence_time", None)
                final_win_rate = metrics.get("final_win_rate", 0) * 100  # Convert to percentage
                
                # Calculate Convergence-Normalized Performance
                # Only if convergence was detected
                if converged and convergence_time:
                    memory_mb = metrics.get("resource_usage", {}).get(agent_name, {}).get("avg_memory_mb", 0)
                    memory_gb = memory_mb / 1024.0
                    conv_time_hrs = convergence_time / 3600.0
                    
                    # CNP for convergence: win_rate / (memory_gb * time_to_converge)
                    if memory_gb > 0 and conv_time_hrs > 0:
                        convergence_cnp = final_win_rate / (memory_gb * conv_time_hrs)
                    else:
                        convergence_cnp = 0
                else:
                    convergence_cnp = 0
                
                cnp_metrics[agent_name].update({
                    "converged": converged,
                    "convergence_episode": convergence_episode,
                    "convergence_time_hrs": convergence_time / 3600.0 if convergence_time else None,
                    "convergence_win_rate": final_win_rate,
                    "convergence_cnp": convergence_cnp
                })
        
        # Save CNP metrics
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cnp_dir = os.path.join(self.log_dir, f"cnp_metrics_{timestamp}")
        os.makedirs(cnp_dir, exist_ok=True)
        
        with open(os.path.join(cnp_dir, "cnp_metrics.json"), 'w') as f:
            json.dump(cnp_metrics, f, indent=2)
            
        print("\n===== Cost-Normalized Performance (CNP) =====")
        for agent_name, metrics in cnp_metrics.items():
            print(f"\nAgent: {agent_name}")
            
            if "convergence_cnp" in metrics:
                if metrics["converged"]:
                    print(f"  Convergence CNP: {metrics['convergence_cnp']:.4f}")
                    print(f"  Convergence Win Rate: {metrics['convergence_win_rate']:.2f}%")
                    print(f"  Convergence Time: {metrics['convergence_time_hrs']:.2f} hours")
                    print(f"  Convergence Episode: {metrics['convergence_episode']}")
                else:
                    print(f"  No convergence detected")
            
            if "fixed_episodes_cnp" in metrics:
                print(f"  Fixed Episodes CNP: {metrics['fixed_episodes_cnp']:.4f}")
                print(f"  Fixed Episodes Win Rate: {metrics['fixed_episodes_win_rate']:.2f}%")
                
            if "fixed_resource_cnp" in metrics:
                print(f"  Fixed Resource CNP: {metrics['fixed_resource_cnp']:.4f}")
                print(f"  Fixed Resource Win Rate: {metrics['fixed_resource_win_rate']:.2f}%")
                print(f"  Fixed Resource Episodes: {metrics['fixed_resource_episodes']}")
        
        return cnp_metrics
    
    def generate_integrated_report(self, convergence_results, fixed_episodes_results, fixed_resource_results):
        """
        Generate a comprehensive integrated evaluation report
        
        Args:
            convergence_results: Results from convergence evaluation
            fixed_episodes_results: Results from fixed episodes evaluation
            fixed_resource_results: Results from fixed resource evaluation
            
        Returns:
            str: Path to the generated report
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.log_dir, f"integrated_evaluation_report_{timestamp}.json")
        
        # Calculate CNP metrics
        cnp_metrics = self.calculate_cnp_metrics(
            fixed_episodes_results,
            fixed_resource_results,
            convergence_results
        )
        
        # Combine all results
        report = {
            "convergence_results": convergence_results,
            "fixed_episodes_results": fixed_episodes_results,
            "fixed_resource_results": fixed_resource_results,
            "cnp_metrics": cnp_metrics,
            "settings": self.settings,
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis": {}
        }
        
        # Add comparative analysis
        agents = list(set(list(convergence_results.keys()) + 
                          list(fixed_episodes_results.keys()) + 
                          list(fixed_resource_results.keys())))
        
        for agent in agents:
            # Compare convergence to fixed episodes
            conv_win_rate = cnp_metrics.get(agent, {}).get("convergence_win_rate", 0)
            ep_win_rate = cnp_metrics.get(agent, {}).get("fixed_episodes_win_rate", 0)
            
            # Compare fixed episodes to fixed resources
            res_win_rate = cnp_metrics.get(agent, {}).get("fixed_resource_win_rate", 0)
            res_episodes = cnp_metrics.get(agent, {}).get("fixed_resource_episodes", 0)
            
            # Calculate various metrics
            report["analysis"][agent] = {
                "convergence_vs_fixed_episodes": conv_win_rate - ep_win_rate,
                "fixed_episodes_vs_fixed_resource": ep_win_rate - res_win_rate,
                "episodes_per_hour": res_episodes / (self.settings["fixed_resource_time"] / 3600),
                "best_cnp": max(
                    cnp_metrics.get(agent, {}).get("convergence_cnp", 0),
                    cnp_metrics.get(agent, {}).get("fixed_episodes_cnp", 0),
                    cnp_metrics.get(agent, {}).get("fixed_resource_cnp", 0)
                )
            }
        
        # Save the report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nIntegrated evaluation report saved to: {output_file}")
        
        # Trigger visualization if available
        try:
            from visualization.cnp_analysis import run_cnp_analysis
            visualization_dir = os.path.join(self.log_dir, f"visualizations_{timestamp}")
            run_cnp_analysis(self.log_dir, visualization_dir)
            print(f"Visualizations created in: {visualization_dir}")
        except ImportError:
            print("Visualization module not available. Skipping visualization generation.")
        
        return output_file


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Integrated Evaluation Framework')
    
    parser.add_argument('--mode', type=str, default='integrated',
                        choices=['integrated', 'convergence', 'fixed-episodes', 'fixed-resource'],
                        help='Evaluation mode (default: integrated)')
    
    parser.add_argument('--agents', nargs='+', default=['random', 'mcts', 'ea', 'marl'],
                       choices=['random', 'mcts', 'ea', 'marl'],
                       help='List of agent strategies to evaluate (default: all)')
    
    parser.add_argument('--log-dir', type=str, default='./evaluations',
                       help='Directory to save evaluation results (default: ./evaluations)')
    
    parser.add_argument('--config-dir', type=str, default=None,
                       help='Directory for configuration files (default: auto-detect)')
    
    parser.add_argument('--difficulty', type=str, default=None,
                       choices=['easy', 'normal', 'hard'],
                       help='Game difficulty (overrides config file)')
    
    return parser.parse_args()


def get_agent_strategies(agent_names):
    """Get strategy tuples for the specified agents"""
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
            print(f"Warning: Unknown agent '{name}' skipped.")
    
    return strategies


def main():
    """Main entry point"""
    args = parse_args()
    
    # Get agent strategies
    agent_strategies = get_agent_strategies(args.agents)
    if not agent_strategies:
        print("Error: No valid agent strategies selected.")
        return
    
    print(f"Selected agents: {[name for _, name in agent_strategies]}")
    
    # Create evaluation framework
    evaluator = IntegratedEvaluationFramework(
        log_dir=args.log_dir,
        config_dir=args.config_dir
    )
    
    # Override difficulty if specified
    if args.difficulty:
        evaluator.settings["difficulty"] = args.difficulty
        print(f"Overriding difficulty: {args.difficulty}")
    
    # Run evaluation based on mode
    if args.mode == 'integrated':
        evaluator.run_integrated_evaluation(agent_strategies)
    elif args.mode == 'convergence':
        evaluator.run_convergence_evaluation(agent_strategies)
    elif args.mode == 'fixed-episodes':
        evaluator.run_fixed_episodes_evaluation(agent_strategies)
    elif args.mode == 'fixed-resource':
        evaluator.run_fixed_resource_evaluation(agent_strategies)
    else:
        print(f"Error: Unknown mode '{args.mode}'")


if __name__ == "__main__":
    main()