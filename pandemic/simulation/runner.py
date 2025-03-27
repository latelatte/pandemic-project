import time
import os
import json
import math
from pandemic.simulation.pandemic import PandemicSimulation
from pandemic.utils.metrics_utils import MetricsCollector
from pandemic.utils.logging_utils import SimulationLogger
from pandemic.utils.resource_utils import ResourceMonitor

class SimulationRunner:
    """class to run the simulation and collect metrics"""
    
    def __init__(self, n_episodes=100, log_dir="./logs", num_players=4, difficulty="normal"):
        """
        Initialize the simulation runner.
        
        Args:
            n_episodes (int): Number of episodes to run.
            log_dir (str): Directory to save logs.
            num_players (int): Number of players in the simulation.
            difficulty (str): Difficulty level of the game.
        """
        self.n_episodes = n_episodes
        self.log_dir = log_dir
        self.num_players = num_players
        self.difficulty = difficulty
        
        self.wins = 0
        self.losses = 0
        
        self._setup_logging()
        self.metrics = None
        self.resource_monitor = ResourceMonitor()
        
    def _setup_logging(self):
        self.logger = SimulationLogger(self.log_dir)

    def run_experiments(self, strategies, config_dir=None):
        agent_names = [name for _, name in strategies]
        self.metrics = MetricsCollector(agent_names)
        
        for ep in range(self.n_episodes):
            sim = PandemicSimulation(*strategies, 
                                   config_dir=config_dir, 
                                   num_players=self.num_players,
                                   difficulty=self.difficulty)
            
            start_time = time.time()
            
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
            
            self.logger.save_episode_log(sim, ep)
            self.logger.log_episode(sim, ep, win, self.metrics.metrics)
            
            end_time = time.time()
            print(f"Episode took {end_time - start_time:.2f} seconds")


            self.evaluate_agent_progress(ep)


        self.logger.log_experiment_summary(
            self.wins, 
            self.n_episodes, 
            self.metrics.get_summary()['avg_turns']
        )
        
        self.logger.close()
        
        metrics_summary = self.metrics.get_summary()
        resource_summary = self.resource_monitor.get_summary()
        
        metrics_data = {
            "avg_turns": metrics_summary["avg_turns"],
            "avg_outbreaks": metrics_summary["avg_outbreaks"],
            "win_rate": self.wins / self.n_episodes,
            "loss_rate": self.losses / self.n_episodes,
            "agent_performance": metrics_summary["agent_performance"],
            "resource_usage": resource_summary
        }
        
        def sanitize_metrics(data):
            """convert NaN and Inf to 0.0"""
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        sanitize_metrics(v)
                    elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        data[k] = 0.0
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, (dict, list)):
                        sanitize_metrics(item)
                    elif isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                        data[i] = 0.0

        sanitize_metrics(metrics_data)

        metrics_file = os.path.join(self.logger.log_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"saves metrics as json: {metrics_file}")
        
        import pandemic.agents.ea_agent as ea
        import pandemic.agents.mcts_agent as mcts
        import pandemic.agents.marl_agent as marl
        
        if hasattr(ea, "_global_ea_agent") and ea._global_ea_agent:
            ea._global_ea_agent.save_state(os.path.join(self.logger.log_dir, "ea_agent_state.pkl"))
        
        if hasattr(mcts, "_global_mcts_agent") and mcts._global_mcts_agent:
            mcts._global_mcts_agent.save_state(os.path.join(self.logger.log_dir, "mcts_agent_state.pkl"))
        
        if hasattr(marl, "_global_marl_agent") and marl._global_marl_agent:
            marl._global_marl_agent.save_state(os.path.join(self.logger.log_dir, "marl_agent_state.pt"))
        
        try:
            for agent_name, _ in strategies:
                if agent_name == "ea":
                    from pandemic.agents.ea_agent import _global_ea_agent
                    if _global_ea_agent:
                        state_file = os.path.join(self.log_dir, "ea_agent_state.pkl")
                        _global_ea_agent.save_state(filename=state_file)
                        print(f"saved state of EA agents: {state_file}")
                
                elif agent_name == "mcts":
                    from pandemic.agents.mcts_agent import _global_mcts_agent
                    if _global_mcts_agent:
                        state_file = os.path.join(self.log_dir, "mcts_agent_state.pkl")
                        _global_mcts_agent.save_state(filepath=state_file)
                        print(f"saved state of MCTS agents: {state_file}")
                
                elif agent_name == "marl":
                    from pandemic.agents.marl_agent import _global_marl_agent
                    if _global_marl_agent:
                        state_file = os.path.join(self.log_dir, "marl_agent_state.pt")
                        _global_marl_agent.save_state(filepath=state_file)
                        print(f"saved state of MARL agents: {state_file}")
        except Exception as e:
            print(f"error in proccessing saving agents state: {e}")
        
        self.print_summary()
        return metrics_data

    def make_timed_strategy(self, orig_strategy, agent_name):
        def timed_wrapper(player):
            self.resource_monitor.start_measurement(agent_name)
            
            start_time = time.time()
            result = orig_strategy(player)
            end_time = time.time()
            elapsed = end_time - start_time
            self.metrics.record_action_time(agent_name, elapsed)
            
            self.resource_monitor.end_measurement(agent_name)
            
            return result
        return timed_wrapper

    def print_summary(self):
        total = self.wins + self.losses
        wrate = 100.0 * self.wins / max(1, total)
        
        print(f"\n===RESULT SUMMARY===")
        print(f"Wins={self.wins}, Losses={self.losses}, Rate={wrate:.2f}%")
        
        metrics_summary = self.metrics.get_summary()
        print(f"Average turns: {metrics_summary['avg_turns']:.2f}")
        print(f"Average outbreaks: {metrics_summary['avg_outbreaks']:.2f}")
        
        print("\n===AGENT PERFORMANCE===")
        for name, data in metrics_summary['agent_performance'].items():
            print(f"{name}: Avg time per move: {data['avg_time_ms']:.2f}ms, " 
                  f"Win contribution: {data['win_contribution']*100/self.n_episodes:.2f}%")
        
        print("\n===RESOURCE USAGE===")
        resource_summary = self.resource_monitor.get_summary()
        for name, data in resource_summary.items():
            print(f"{name}: Avg Memory: {data['avg_memory_mb']:.2f}MB, " 
                  f"Avg CPU: {data['avg_cpu_percent']:.2f}%")

    def evaluate_agent_progress(self, current_episode):
        """Evaluate the agent's progress and log the results."""
        if current_episode % 10 != 0:  # by default, every 10 episodes
            return
            
        recent_win_rate = self.wins / max(1, (self.wins + self.losses))
        
        self.logger.writer.add_scalar('Evaluation/WinRate', recent_win_rate * 100, current_episode)
        
        metrics_summary = self.metrics.get_summary()
        for agent_name, data in metrics_summary['agent_performance'].items():

            self.logger.writer.add_scalar(f'Evaluation/{agent_name}/AvgTime', 
                                         data['avg_time_ms'], current_episode)
            
            win_contribution = data['win_contribution'] / max(1, current_episode)
            self.logger.writer.add_scalar(f'Evaluation/{agent_name}/WinContribution', 
                                         win_contribution * 100, current_episode)


if __name__ == "__main__":

    runner = SimulationRunner(n_episodes=5)

    print("Done.")