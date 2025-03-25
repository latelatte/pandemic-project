import random
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
    """進化的アルゴリズムを使用するエージェント"""
    def __init__(self, name="EA", population_size=50, generations=10):
        super().__init__(name)
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_individual = None
        
        # ゲノムパラメータの初期化
        self.genome_length = 20  # アクション優先度の数
        self.initialize_population()
        
    def initialize_population(self):
        """初期集団を生成"""
        self.population = [
            Individual(genome_length=self.genome_length) 
            for _ in range(self.population_size)
        ]
        
    def evaluate_fitness(self, individual, player, simulation):
        """個体の適応度（ゲームで勝つ確率）を評価"""
        # 実際には複数回シミュレーションを実行して勝率を測定
        # ここでは簡易実装としてゲノムから直接評価
        
        # ゲノムを使ってアクション優先度を決定
        action_priorities = {}
        
        # 移動アクション（隣接都市への移動の優先度）
        for i, neighbor in enumerate(player.city.neighbours[:4]):  # 最大4都市まで
            idx = i % len(individual.genome)
            action_priorities[f"move_{neighbor.name}"] = individual.genome[idx]
        
        # 治療アクション（感染レベルが高いほど優先）
        idx = 4 % len(individual.genome)
        action_priorities["treat"] = individual.genome[idx] * player.city.infection_level
        
        # 研究所建設アクション
        idx = 5 % len(individual.genome)
        has_station = 1 if player.city.has_research_station else 0
        action_priorities["build"] = individual.genome[idx] * (1 - has_station)
        
        # 現在の状態から推定した勝率
        # 感染レベルが低いほど、研究所が多いほど勝率が高いと仮定
        total_infection = sum(city.infection_level for city in simulation.cities)
        total_stations = sum(1 for city in simulation.cities if city.has_research_station)
        
        # 感染が0に近く、研究所が多いほど適応度が高い
        fitness = 1.0 - (total_infection / (len(simulation.cities) * 5))
        fitness += 0.2 * (total_stations / len(simulation.cities))
        
        individual.fitness = max(0, min(1, fitness))
        return individual.fitness
        
    def select_parents(self, tournament_size=3):
        """トーナメント選択で親を選ぶ"""
        participants = random.sample(self.population, tournament_size)
        return max(participants, key=lambda ind: ind.fitness)
        
    def evolve(self, player, simulation):
        """複数世代の進化計算を実行"""
        # 現在の集団の適応度を評価
        for individual in self.population:
            self.evaluate_fitness(individual, player, simulation)
            
        for generation in range(self.generations):
            new_population = []
            
            # エリート保存: 最良の個体をそのまま次世代に
            elite = max(self.population, key=lambda ind: ind.fitness)
            new_population.append(Individual(genome=elite.genome.copy()))
            
            # 残りは選択・交叉・突然変異で生成
            while len(new_population) < self.population_size:
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                
                # 交叉と突然変異
                child = parent1.crossover(parent2)
                child.mutate()
                
                new_population.append(child)
                
            # 集団の更新
            self.population = new_population
            
            # 新集団の適応度評価
            for individual in self.population:
                self.evaluate_fitness(individual, player, simulation)
                
        # 最良個体を記録
        self.best_individual = max(self.population, key=lambda ind: ind.fitness)
        
    def decide_action(self, player, simulation):
        """進化計算を使って最適なアクションを決定"""
        # 進化計算で最良個体を探索
        self.evolve(player, simulation)
        
        # 最良個体のゲノムを使ってアクションを決定
        possible_actions = self._get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
            
        # ゲノムでアクションに優先度をつける
        action_scores = []
        for i, action in enumerate(possible_actions):
            idx = i % len(self.best_individual.genome)
            score = self.best_individual.genome[idx]
            
            # アクションのタイプによる調整
            if action["type"] == "treat" and player.city.infection_level > 0:
                score *= 1.2  # 治療アクションの優先度上昇
                
            action_scores.append((action, score))
            
        # 最も優先度の高いアクションを選択
        best_action = max(action_scores, key=lambda x: x[1])[0]
        
        self.record_action("ea_decision", {
            "action": best_action,
            "fitness": self.best_individual.fitness
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

