import time
import psutil
import os
import numpy as np  # 平均値計算に使用

import time
import psutil
import os
import numpy as np

class ResourceMonitor:
    """resources usage monitor for agents"""
    
    _instance = None
    """Singleton instance of ResourceMonitor"""
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceMonitor, cls).__new__(cls)
            cls._instance.process = psutil.Process(os.getpid())
            cls._instance.measurements = {}
        
        known_agents = ["MCTSAgent", "EAAgent", "MARLAgent"]
        for agent in known_agents:
            cls._instance.measurements[agent] = {
                'memory': [],
                'cpu_percent': [],
                'timestamps': []
            }
        return cls._instance
        
    def start_measurement(self, agent_name):
        if agent_name not in self.measurements:
            self.measurements[agent_name] = {
                'memory': [],
                'cpu_percent': [],
                'timestamps': []
            }
        
        # Take multiple samples before establishing baseline
        memory_samples = []
        for _ in range(3):  # Take 3 samples
            memory_samples.append(self.process.memory_info().rss / 1024 / 1024)
            time.sleep(0.01)  # Short delay between samples
            
        self.baseline_memory = sum(memory_samples) / len(memory_samples)  # Average baseline
        self.start_time = time.time()
        
    def end_measurement(self, agent_name):
        # Take multiple samples for more accurate measurement
        memory_samples = []
        for _ in range(3):  # Take 3 samples
            memory_samples.append(self.process.memory_info().rss / 1024 / 1024)
            time.sleep(0.01)  # Short delay between samples
            
        current_memory = sum(memory_samples) / len(memory_samples)
        memory_usage = max(0.1, current_memory - self.baseline_memory)  # Ensure at least 0.1MB is reported
        
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
            print(f"{agent_name}が見つからないからNone!")
            return None
            
        data = self.measurements[agent_name]
    
        if not data['memory']:
            print(f"memoryのデータがないからNone!")
            return None
        elif not data['cpu_percent']:
            print(f"cpuのデータがないからNone!")
            return None
        elif not data['timestamps']:
            print(f"timestampがないからNone!")
            return None
            
        # Filter out any anomalous zero memory readings
        filtered_memory = [m for m in data['memory'] if m > 0.05]
        if not filtered_memory:
            filtered_memory = data['memory']  # If all readings are near zero, use original data
            
        return {
            "avg_memory_mb": sum(filtered_memory) / len(filtered_memory),
            "avg_cpu_percent": sum(data['cpu_percent']) / len(data['cpu_percent']),
            "avg_time_sec": sum(data['timestamps']) / len(data['timestamps'])
        }
    
    def get_summary(self):
        summary = {}
        
        for agent_name, data in self.measurements.items():
            if data['memory'] and data['cpu_percent'] and data['timestamps']:
                # Filter out any anomalous zero memory readings
                filtered_memory = [m for m in data['memory'] if m > 0.05]
                if not filtered_memory:
                    filtered_memory = data['memory']
                    
                summary[agent_name] = {
                    "avg_memory_mb": sum(filtered_memory) / len(filtered_memory),
                    "avg_cpu_percent": sum(data['cpu_percent']) / len(data['cpu_percent']),
                    "avg_time_sec": sum(data['timestamps']) / len(data['timestamps'])
                }
            else:
                # default values if no measurements were taken
                summary[agent_name] = {
                    "avg_memory_mb": 0.1,  # minimum reportable value
                    "avg_cpu_percent": 0.0,
                    "avg_time_sec": 0.0
                }
        
        return summary
        
    def get_current_memory_usage(self, agent_name):
        if agent_name not in self.measurements or not self.measurements[agent_name]['memory']:
            return 0.1
            
        recent_memory = self.measurements[agent_name]['memory'][-10:]
        if not recent_memory:
            return 0.1
        
        filtered_memory = [m for m in recent_memory if m > 0.05]
        if not filtered_memory:
            filtered_memory = recent_memory
            
        return sum(filtered_memory) / len(filtered_memory)
    
    def print_stats(self):
        """display the resource usage statistics"""
        print("\n== resouce used ==")
        summary = self.get_summary()
        
        for agent_name, stats in summary.items():
            print(f"Agent: {agent_name}")
            print(f"  avg memory: {stats['avg_memory_mb']:.2f} MB")
            print(f"  avg CPU: {stats['avg_cpu_percent']:.2f}%")
            print(f"  avg time: {stats['avg_time_sec'] * 1000:.2f} ms")