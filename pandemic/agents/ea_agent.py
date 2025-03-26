import random
from deap import base, creator, tools, algorithms
import numpy as np
from pandemic.agents.baseline_agents import BaseAgent

class Individual:
    """遺伝的アルゴリズムの個体クラス"""
    def __init__(self, genome_length=20, genome=None):
        if genome is None:
            # ランダム初期化: ゲノムは行動選択の優先度を表す
            self.genome = np.random.random(genome_length)
        else:
            self.genome = genome
        self.fitness = 0.0
        
    def mutate(self, mutation_rate=0.1):
        """突然変異: 一部の遺伝子をランダムに変更"""
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] = random.random()
        return self
    
    def crossover(self, other, crossover_rate=0.7):
        """交叉: 二つの個体のゲノムを組み合わせた子を生成"""
        if random.random() > crossover_rate:
            return Individual(genome=self.genome.copy())
            
        # 一点交叉
        crossover_point = random.randint(0, len(self.genome) - 1)
        child_genome = np.concatenate([
            self.genome[:crossover_point],
            other.genome[crossover_point:]
        ])
        
        return Individual(genome=child_genome)

class EAAgent(BaseAgent):
    """DEAPを使った進化的アルゴリズムエージェント"""
    def __init__(self, name="DEAP-EA", population_size=50, generations=10):
        super().__init__(name)
        self.population_size = population_size
        self.generations = generations
        self.genome_length = 20
        
        # DEAP初期設定
        self._setup_deap()
        
    def _setup_deap(self):
        """DEAPのセットアップ"""
        # 最大化問題の定義
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # 個体の定義
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # ツールボックスの初期化
        self.toolbox = base.Toolbox()
        # 遺伝子の生成関数
        self.toolbox.register("attr_float", random.random)
        # 個体の生成
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_float, self.genome_length)
        # 集団の生成
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 遺伝的操作の設定
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def evaluate_fitness(self, individual, player, simulation):
        """適応度評価関数"""
        # 省略: 既存のコードと同様のロジック
        # ...
        
        total_infection = sum(city.infection_level for city in simulation.cities)
        total_stations = sum(1 for city in simulation.cities if city.has_research_station)
        
        fitness = 1.0 - (total_infection / (len(simulation.cities) * 5))
        fitness += 0.2 * (total_stations / len(simulation.cities))
        
        return (max(0, min(1, fitness)),)  # DEAPは適応度をタプルで扱う
    
    def decide_action(self, player, simulation):
        """DEAPを使って最適なアクションを決定"""
        # 評価関数をプレイヤーとシミュレーションに結びつける
        self.toolbox.register("evaluate", self.evaluate_fitness, 
                             player=player, simulation=simulation)
        
        # 初期集団を生成
        pop = self.toolbox.population(n=self.population_size)
        
        # 統計情報を設定
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        # 進化アルゴリズムの実行（eaSimpleはDEAP提供の標準的な遺伝的アルゴリズム）
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox, cxpb=0.7, mutpb=0.2, 
            ngen=self.generations, stats=stats, verbose=False)
        
        # 最良個体の取得
        best_ind = tools.selBest(pop, 1)[0]
        
        # 最良個体のゲノムを使ってアクションを選択（既存のロジック）
        possible_actions = self._get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
            
        # ゲノムでアクションに優先度をつける
        action_scores = []
        for i, action in enumerate(possible_actions):
            idx = i % len(best_ind)
            score = best_ind[idx]
            
            # アクションのタイプによる調整
            if action["type"] == "treat" and player.city.infection_level > 0:
                score *= 1.2  # 治療アクションの優先度上昇
                
            action_scores.append((action, score))
            
        # 最も優先度の高いアクションを選択
        best_action = max(action_scores, key=lambda x: x[1])[0]
        
        self.record_action("ea_decision", {
            "action": best_action,
            "fitness": best_ind.fitness.values[0]
        })
        
        return best_action

    def _get_possible_actions(self, player, simulation):
        """可能なアクションリストを取得（ベースエージェントと同様）"""
        actions = []
        
        # 移動アクション
        for neighbor in player.city.neighbours:
            actions.append({
                "type": "move",
                "target_city": neighbor
            })
            
        # 治療アクション
        if player.city.infection_level > 0:
            actions.append({
                "type": "treat",
                "city": player.city
            })
            
        # その他のアクション
        
        return actions

# EA戦略関数
def ea_agent_strategy(player):
    agent = EAAgent()
    while player.actions_remaining > 0:
        action = agent.decide_action(player, player.simulation)
        if not action:
            break
            
        # アクション実行
        # ...アクションのタイプに応じた処理
        
        player.actions_remaining -= 1

