import time
from pandemic.simulation.pandemic import PandemicSimulation
from pandemic.utils.metrics_utils import MetricsCollector
from pandemic.utils.logging_utils import SimulationLogger
from pandemic.utils.resource_utils import ResourceMonitor

class SimulationRunner:
    def __init__(self, n_episodes=10, log_dir="./logs"):
        self.n_episodes = n_episodes
        self.wins = 0
        self.losses = 0
        
        # ユーティリティクラスのインスタンス化
        self.logger = SimulationLogger(log_dir)
        self.metrics = MetricsCollector()
        self.resource_monitor = ResourceMonitor()

    def run_experiments(self, strategies):
        """複数のエージェント戦略を評価"""
        # メトリクスコレクターの初期化
        agent_names = [name for _, name in strategies]
        self.metrics = MetricsCollector(agent_names)
        
        for ep in range(self.n_episodes):
            sim = PandemicSimulation(*strategies)
            
            # 実行時間の計測開始
            start_time = time.time()
            
            # 各エージェントの行動時間を記録するためのモニター設定
            for p in sim.players:
                original_strategy = p.strategy
                strategy_name = p.strategy_name
                
                # クロージャ変数を適切にキャプチャするように定義
                def make_timed_strategy(player, orig_strategy, agent_name):
                    def timed_strategy():
                        # リソース計測開始
                        self.resource_monitor.start_measurement(agent_name)  # 正しい変数を使用
                        
                        # 行動時間計測
                        move_start = time.time()
                        orig_strategy()  # プレイヤーを渡す
                        move_end = time.time()
                        
                        # リソース計測終了
                        self.resource_monitor.end_measurement(agent_name)  # 正しい変数を使用
                        
                        # 時間記録
                        self.metrics.record_action_time(agent_name, move_end - move_start)
                    
                    return timed_strategy
                
                # 各プレイヤー専用の戦略関数を生成
                p.strategy = make_timed_strategy(p, original_strategy, strategy_name)
            
            sim.run_game()
            
            # 勝敗判定
            win = all(c.infection_level == 0 for c in sim.cities)
            if win:
                print(f"Episode {ep+1}: WIN.")
                self.wins += 1
            else:
                print(f"Episode {ep+1}: LOSE.")
                self.losses += 1
            
            # メトリクス収集
            self.metrics.record_game_metrics(sim, win)
            
            # ログ記録
            self.logger.save_episode_log(sim, ep)
            self.logger.log_episode(sim, ep, win, self.metrics.metrics)
            
            # 実行時間の計測終了
            end_time = time.time()
            print(f"Episode took {end_time - start_time:.2f} seconds")

        # 実験サマリーをログに記録
        self.logger.log_experiment_summary(
            self.wins, 
            self.n_episodes, 
            self.metrics.get_summary()['avg_turns']
        )
        
        # TensorBoardリソース解放
        self.logger.close()
        
        self.print_summary()

    def print_summary(self):
        """詳細な結果サマリーを出力"""
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

### メイン関数例
if __name__ == "__main__":

    runner = SimulationRunner(n_episodes=5)
    runner.run_experiments(strategies)

    print("Done.")