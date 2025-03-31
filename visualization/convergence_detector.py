import time

class ConvergenceDetector:
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
        self.win_history.append(1 if win else 0)
        
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