import time
import psutil
import os
import numpy as np  # 平均値計算に使用

class ResourceMonitor:
    """resources usage monitor for agents"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.measurements = {}
        
    def start_measurement(self, agent_name):
        if agent_name not in self.measurements:
            self.measurements[agent_name] = {
                'memory': [],
                'cpu_percent': [],
                'timestamps': []
            }
        
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
        
    def end_measurement(self, agent_name):
        memory_usage = self.process.memory_info().rss / 1024 / 1024 - self.baseline_memory
        cpu_percent = self.process.cpu_percent()
        elapsed = time.time() - self.start_time
        
        self.measurements[agent_name]['memory'].append(memory_usage)
        self.measurements[agent_name]['cpu_percent'].append(cpu_percent)
        self.measurements[agent_name]['timestamps'].append(elapsed)
        
        return {
            'memory_mb': memory_usage,
            'cpu_percent': cpu_percent,
            'time_sec': elapsed
        }
    
    def get_agent_summary(self, agent_name):
        if agent_name not in self.measurements:
            return None
            
        data = self.measurements[agent_name]
        
        if not data['memory'] or not data['cpu_percent'] or not data['timestamps']:
            return None
            
        return {
            "avg_memory_mb": sum(data['memory']) / len(data['memory']),
            "avg_cpu_percent": sum(data['cpu_percent']) / len(data['cpu_percent']),
            "avg_time_sec": sum(data['timestamps']) / len(data['timestamps'])
        }
    
    def get_summary(self):
        summary = {}
        
        for agent_name, data in self.measurements.items():
            if data['memory'] and data['cpu_percent'] and data['timestamps']:
                summary[agent_name] = {
                    "avg_memory_mb": sum(data['memory']) / len(data['memory']),
                    "avg_cpu_percent": sum(data['cpu_percent']) / len(data['cpu_percent']),
                    "avg_time_sec": sum(data['timestamps']) / len(data['timestamps'])
                }
            else:
                # default values if no measurements were taken
                summary[agent_name] = {
                    "avg_memory_mb": 0.0,
                    "avg_cpu_percent": 0.0,
                    "avg_time_sec": 0.0
                }
        
        return summary
        
    def print_stats(self):
        """display the resource usage statistics"""
        print("\n== resouce used ==")
        summary = self.get_summary()
        
        for agent_name, stats in summary.items():
            print(f"Agent: {agent_name}")
            print(f"  avg memory: {stats['avg_memory_mb']:.2f} MB")
            print(f"  avg CPU: {stats['avg_cpu_percent']:.2f}%")
            print(f"  avg time: {stats['avg_time_sec'] * 1000:.2f} ms")