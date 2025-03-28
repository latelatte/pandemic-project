"""
Dual Evaluation Framework: Module implementing both fixed-episode and fixed-resource evaluation
approaches for agent performance assessment
"""

import time
import os
import datetime
import json
import psutil
import threading
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pandemic.simulation.pandemic import PandemicSimulation
from pandemic.simulation.runner import SimulationRunner
from pandemic.utils.metrics_utils import MetricsCollector
from pandemic.utils.resource_utils import ResourceMonitor

class DualEvaluationFramework:
    """
    Dual evaluation framework for AI agents assessment:
    1. Fixed-Episodes Evaluation: Run all agents for the same number of episodes
    2. Fixed-Resource Evaluation: Allocate the same computational resources to all agents
    """
    
    def __init__(self, log_dir="./evaluations", config_dir=None):
        """
        Initialize the evaluation framework
        
        Args:
            log_dir: Directory to save evaluation results
            config_dir: Directory for configuration files
        """
        self.log_dir = log_dir
        
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
        self.fixed_episodes_results = {}
        self.fixed_resource_results = {}
        
        # Default settings
        self.default_episodes = 5000  # As specified in the paper
        self.default_resource_time = 24 * 60 * 60  # 24 hours in seconds
        self.eval_episodes = 1000  # Number of episodes for evaluation
        
        # Load settings file
        self.settings = self._load_evaluation_config()
        
    def _load_evaluation_config(self):
        """Load evaluation settings or use defaults"""
        settings_file = os.path.join(self.config_dir, "evaluation_config.json")
        default_settings = {
            "fixed_episodes": self.default_episodes,
            "fixed_resource_time": self.default_resource_time,
            "eval_episodes": self.eval_episodes,
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
    
    def run_fixed_episodes_evaluation(self, agent_strategies, episodes=None):
        """
        Run evaluation with a fixed number of episodes
        
        Args:
            agent_strategies: List of agent strategy tuples [(func, name), ...]
            episodes: Number of episodes to run (None for default)
            
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
        
        print(f"\nFixed Episodes Evaluation completed in {total_time:.2f} seconds")
        
        # Save overall results
        summary = {
            "evaluation_type": "fixed_episodes",
            "episodes": episodes,
            "total_time": total_time,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(os.path.join(eval_dir, "evaluation_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.fixed_episodes_results = results
        return results
    
    def run_fixed_resource_evaluation(self, agent_strategies, resource_time=None):
        """
        Run evaluation with fixed computational resources (time-based)
        
        Args:
            agent_strategies: List of agent strategy tuples [(func, name), ...]
            resource_time: Time limit in seconds (None for default)
            
        Returns:
            dict: Evaluation results
        """
        if resource_time is None:
            resource_time = self.settings["fixed_resource_time"]
            
        print(f"\n===== Starting Fixed Resource Evaluation: {resource_time} seconds =====")
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
            episodes_completed = agent_metrics.get("episodes_completed", 0)
            
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Episodes Completed: {episodes_completed}")
            print(f"  Avg Decision Time: {avg_time:.2f} ms")
            print(f"  Avg Memory Usage: {memory_mb:.2f} MB")
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print(f"\nFixed Resource Evaluation completed in {total_time:.2f} seconds")
        
        # Save overall results
        summary = {
            "evaluation_type": "fixed_resource",
            "resource_time": resource_time,
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
                    
                    # Update episode count for reporting
                    self.episodes_completed = ep + 1
                
                metrics_summary = self.metrics.get_summary()
                resource_summary = self.resource_monitor.get_summary()
                
                metrics_data = {
                    "win_rate": self.metrics.metrics.get('win_rates', {}).get(strategy_name, 0) / max(1, self.episodes_completed),
                    "episodes_completed": self.episodes_completed,
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
        metrics["episodes_completed"] = getattr(runner, "episodes_completed", 0)
        
        return metrics
    
    def run_complete_evaluation(self, agent_strategies):
        """
        Run both fixed-episodes and fixed-resource evaluations
        
        Args:
            agent_strategies: List of agent strategy tuples [(func, name), ...]
            
        Returns:
            tuple: (fixed_episodes_results, fixed_resource_results)
        """
        print("\n========== STARTING COMPLETE DUAL EVALUATION ==========")
        
        # First run fixed-episodes evaluation
        fixed_episodes_results = self.run_fixed_episodes_evaluation(agent_strategies)
        
        # Then run fixed-resource evaluation
        fixed_resource_results = self.run_fixed_resource_evaluation(agent_strategies)
        
        # Calculate CNP (Cost-Normalized Performance) for each agent
        self.calculate_cnp_metrics(fixed_episodes_results, fixed_resource_results)
        
        print("\n========== DUAL EVALUATION COMPLETED ==========")
        
        return fixed_episodes_results, fixed_resource_results
    
    def calculate_cnp_metrics(self, fixed_episodes_results=None, fixed_resource_results=None):
        """
        Calculate CNP (Cost-Normalized Performance) metrics
        
        Args:
            fixed_episodes_results: Results from fixed episodes evaluation
            fixed_resource_results: Results from fixed resource evaluation
            
        Returns:
            dict: CNP metrics
        """
        if fixed_episodes_results is None:
            fixed_episodes_results = self.fixed_episodes_results
            
        if fixed_resource_results is None:
            fixed_resource_results = self.fixed_resource_results
            
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
            episodes = metrics.get("episodes_completed", 0)
            
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
        
        # Save CNP metrics
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cnp_dir = os.path.join(self.log_dir, f"cnp_metrics_{timestamp}")
        os.makedirs(cnp_dir, exist_ok=True)
        
        with open(os.path.join(cnp_dir, "cnp_metrics.json"), 'w') as f:
            json.dump(cnp_metrics, f, indent=2)
            
        print("\n===== Cost-Normalized Performance (CNP) =====")
        for agent_name, metrics in cnp_metrics.items():
            print(f"\nAgent: {agent_name}")
            
            if "fixed_episodes_cnp" in metrics:
                print(f"  Fixed Episodes CNP: {metrics['fixed_episodes_cnp']:.4f}")
                print(f"  Fixed Episodes Win Rate: {metrics['fixed_episodes_win_rate']:.2f}%")
                
            if "fixed_resource_cnp" in metrics:
                print(f"  Fixed Resource CNP: {metrics['fixed_resource_cnp']:.4f}")
                print(f"  Fixed Resource Win Rate: {metrics['fixed_resource_win_rate']:.2f}%")
                print(f"  Fixed Resource Episodes: {metrics['fixed_resource_episodes']}")
        
        return cnp_metrics
    
    def generate_evaluation_report(self, output_file=None):
        """
        Generate a comprehensive evaluation report with comparisons
        
        Args:
            output_file: Path to save the report (None for auto-generation)
            
        Returns:
            str: Path to the generated report
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"evaluation_report_{timestamp}.json")
            
        # Combine all results
        report = {
            "fixed_episodes_results": self.fixed_episodes_results,
            "fixed_resource_results": self.fixed_resource_results,
            "cnp_metrics": self.calculate_cnp_metrics(),
            "settings": self.settings,
            "timestamp": datetime.datetime.now().isoformat(),
            "comparison_summary": {}
        }
        
        # Add comparison summary
        if self.fixed_episodes_results and self.fixed_resource_results:
            agents = list(set(list(self.fixed_episodes_results.keys()) + 
                             list(self.fixed_resource_results.keys())))
            
            for agent in agents:
                ep_win_rate = self.fixed_episodes_results.get(agent, {}).get("win_rate", 0) * 100
                res_win_rate = self.fixed_resource_results.get(agent, {}).get("win_rate", 0) * 100
                
                ep_episodes = self.settings["fixed_episodes"]
                res_episodes = self.fixed_resource_results.get(agent, {}).get("episodes_completed", 0)
                
                ep_cnp = report["cnp_metrics"].get(agent, {}).get("fixed_episodes_cnp", 0)
                res_cnp = report["cnp_metrics"].get(agent, {}).get("fixed_resource_cnp", 0)
                
                report["comparison_summary"][agent] = {
                    "win_rate_difference": res_win_rate - ep_win_rate,
                    "episodes_ratio": res_episodes / max(1, ep_episodes),
                    "cnp_ratio": res_cnp / max(1, ep_cnp) if ep_cnp > 0 else float('inf')
                }
        
        # Save the report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nEvaluation report saved to: {output_file}")
        return output_file


def run_evaluation(agent_strategies=None, mode="both", episodes=None, resource_time=None):
    """
    Convenience function to run the evaluation framework
    
    Args:
        agent_strategies: List of agent strategy tuples [(func, name), ...]
        mode: Evaluation mode ("episodes", "resource", or "both")
        episodes: Number of episodes for fixed-episodes evaluation
        resource_time: Time limit for fixed-resource evaluation
        
    Returns:
        tuple or dict: Evaluation results based on mode
    """
    
    # Create and run the evaluation framework
    evaluator = DualEvaluationFramework()
    
    if mode == "episodes":
        return evaluator.run_fixed_episodes_evaluation(agent_strategies, episodes)
    elif mode == "resource":
        return evaluator.run_fixed_resource_evaluation(agent_strategies, resource_time)
    else:  # "both" or any other value
        return evaluator.run_complete_evaluation(agent_strategies)


if __name__ == "__main__":
    # Example usage
    from pandemic.agents.baseline_agents import random_agent_strategy
    from pandemic.agents.mcts_agent import mcts_agent_strategy
    from pandemic.agents.ea_agent import ea_agent_strategy
    
    # Define strategies to evaluate
    strategies = [
        (random_agent_strategy, "Random"),
        (mcts_agent_strategy, "MCTS"),
        (ea_agent_strategy, "EA")
    ]
    
    # Run evaluation with default settings
    fixed_episodes_results, fixed_resource_results = run_evaluation(strategies)
    
    print("Evaluation completed successfully!")