import time
import numpy as np

class MetricsCollector:
    """collects and summarizes game metrics"""
    
    def __init__(self, agent_names=None):
        self.metrics = {
            'total_turns': [],
            'outbreak_counts': [],
            'treatment_counts': {},
            'win_rates': {},
            'avg_infection_level': [],
            'time_per_move': {},
            'resource_usage': {}
        }
        
        self.agent_stats = {}
        
        if agent_names:
            for name in agent_names:
                self.metrics['treatment_counts'][name] = 0
                self.metrics['win_rates'][name] = 0
                self.metrics['time_per_move'][name] = []
                self.agent_stats[name] = {
                    "total_time": 0.0,
                    "calls": 0,
                    "win_rates": 0,
                    "treatments": 0
                }

    def record_game_metrics(self, simulation, win=False):
        self.metrics['total_turns'].append(simulation.turn_count)
        self.metrics['outbreak_counts'].append(simulation.outbreak_count)
        self.metrics['avg_infection_level'].append(
            sum(c.infection_level for c in simulation.cities) / len(simulation.cities)
        )
        
        if win:
            for strategy_name in set(p.strategy_name for p in simulation.players):    
                self.metrics['win_rates'][strategy_name] += 1
    
    def record_action_time(self, agent_name, time_taken):
        """records the time taken for an agent to make a move"""
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {
                "total_time": 0.0,
                "calls": 0,
                "win_rates": 0,
                "treatments": 0
            }

        self.agent_stats[agent_name]["total_time"] += time_taken
        self.agent_stats[agent_name]["calls"] += 1
    
    def record_treatment(self, agent_name, count=1):
        """records the number of treatments performed by an agent"""
        self.metrics['treatment_counts'][agent_name] += count
    
    def get_agent_current_stats(self, agent_name):
        if agent_name not in self.agent_stats:
            return {"avg_time_ms": 0.0, "win_rates": 0, "treatments": 0, "avg_turns": 0, "avg_outbreaks": 0}
            
        stats = self.agent_stats[agent_name]
        calls = max(1, stats.get("calls", 0))
        avg_time = stats.get("total_time", 0) / calls
        
        recent_turns = self.metrics['total_turns'][-20:] if self.metrics['total_turns'] else [0]
        recent_outbreaks = self.metrics['outbreak_counts'][-20:] if self.metrics['outbreak_counts'] else [0]
        
        return {
            "avg_time_ms": float(avg_time * 1000.0),
            "win_rates": stats.get("win_rates", 0),
            "treatments": stats.get("treatments", 0),
            "avg_turns": float(np.mean(recent_turns)),
            "avg_outbreaks": float(np.mean(recent_outbreaks))
        }
    
    def get_summary(self):
        summary = {
            'avg_turns': 0.0,
            'avg_outbreaks': 0.0,
            'win_rate': 0.0,
            'loss_rate': 1.0,
            'agent_performance': {},
            'resource_usage': {}
        }
        
        if self.metrics['total_turns']:
            summary['avg_turns'] = float(np.mean(self.metrics['total_turns']))
        if self.metrics['outbreak_counts']:
            summary['avg_outbreaks'] = float(np.mean(self.metrics['outbreak_counts']))
        
        for agent_name, stats in self.agent_stats.items():
            calls = max(1, stats.get("calls", 0)) # Avoid division by zero
            avg_time = stats.get("total_time", 0) / calls
            
            summary['agent_performance'][agent_name] = {
                "avg_time_ms": float(avg_time * 1000.0),  # convert to ms
                "win_rates": stats.get("win_rates", 0),
                "treatments": stats.get("treatments", 0)
            }
        
        return summary

