import time
import psutil
import os

class ResourceMonitor:
    """エージェントのリソース使用量を計測するクラス"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.measurements = {}
        
    def start_measurement(self, agent_name):
        """計測開始前の状態を記録"""
        if agent_name not in self.measurements:
            self.measurements[agent_name] = {
                'memory': [],
                'cpu_percent': [],
                'timestamps': []
            }
        
        # 現在のベースライン状態を記録
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
        
    def end_measurement(self, agent_name):
        """エージェント実行後のリソース状態を記録"""
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
        """特定のエージェントの計測結果サマリーを返す"""
        if agent_name not in self.measurements:
            return {"error": "No measurements for this agent"}
            
        data = self.measurements[agent_name]
        return {
            'peak_memory_mb': max(data['memory']) if data['memory'] else 0,
            'avg_memory_mb': sum(data['memory']) / len(data['memory']) if data['memory'] else 0,
            'peak_cpu_percent': max(data['cpu_percent']) if data['cpu_percent'] else 0,
            'avg_cpu_percent': sum(data['cpu_percent']) / len(data['cpu_percent']) if data['cpu_percent'] else 0,
        }
    
    def get_summary(self):
        """全エージェントの計測結果サマリーを返す"""
        return {name: self.get_agent_summary(name) for name in self.measurements}