import random
import numpy as np
from deap import base, creator, tools, algorithms
import os
import pickle
from pandemic.agents.baseline_agents import BaseAgent

class EAAgent(BaseAgent):
    def __init__(self, name="-EA", population_size=50, generations=10, 
                genome_length=40, crossover_rate=0.7, mutation_rate=0.2,
                dynamic_params=True):
        super().__init__(name)
        self.population_size = population_size
        self.generations = generations
        self.genome_length = genome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.dynamic_params = dynamic_params 
        
        self.best_population = []
        self.best_individual = None
        self.generation_count = 0
        
        self.fitness_history = []
        self.success_counter = {} 

        self._setup_deap()
    
    def _setup_deap(self):
        if 'FitnessMax' in creator.__dict__:
            del creator.FitnessMax
        if 'Individual' in creator.__dict__:
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_float, self.genome_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", self._adaptive_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _adaptive_mutation(self, individual):
        if self.dynamic_params:
            current_rate = max(0.05, self.mutation_rate * (1 - self.generation_count / 100))
        else:
            current_rate = self.mutation_rate
            
        for i in range(len(individual)):
            if random.random() < current_rate:
                # using Gaussian mutation
                individual[i] += random.gauss(0, 0.2)
                individual[i] = max(0.0, min(1.0, individual[i]))
                
        return individual,
    
    def evaluate_fitness(self, individual, player, simulation):
        if simulation.game_over:
            if simulation.is_win_condition():
                return (1.0,)
            return (0.0,)
        
        score = 0.5  # base score
        
        total_infection = sum(city.infection_level for city in simulation.cities)
        max_possible_infection = 3 * len(simulation.cities)
        infection_control = 1.0 - (total_infection / max_possible_infection)
        score += 0.25 * infection_control
        
        research_stations = [city for city in simulation.cities if city.has_research_station]
        station_coverage = min(1.0, len(research_stations) / 4.0)
        score += 0.15 * station_coverage

        cures_discovered = len(getattr(simulation, 'discovered_cures', []))
        cure_progress = cures_discovered / 4.0
        score += 0.40 * cure_progress
        
        outbreak_count = simulation.outbreak_count
        outbreak_limit = simulation.outbreak_limit
        outbreak_safety = 1.0 - (outbreak_count / outbreak_limit)
        score += 0.20 * outbreak_safety
        
        return (max(0.1, min(1, score)),)
    
    def _get_possible_actions(self, player, simulation):
        actions = []
        
        # move
        for neighbor in player.city.neighbours:
            actions.append({
                "type": "move",
                "target_city": neighbor
            })
            
        # treat
        if player.city.infection_level > 0:
            actions.append({
                "type": "treat",
                "city": player.city
            })
            
        # build
        has_city_card = any(
            (hasattr(card, 'city_name') and card.city_name == player.city.name) or
            (hasattr(card, 'city') and hasattr(card.city, 'name') and card.city.name == player.city.name)
            for card in player.hand
        )
        if not player.city.has_research_station and has_city_card:
            actions.append({
                "type": "build",
                "city": player.city
            })
        
        # baseline_agents.pyのget_possible_actions内で治療薬発見のロジックを強化
        cards_by_color = {}
        for card in player.hand:
            if hasattr(card, 'color') and card.color and card.color != "INF":
                color = card.color
                if color not in cards_by_color:
                    cards_by_color[color] = []
                cards_by_color[color].append(card)
        
        return actions
    
    def _score_action(self, action, individual, player, simulation):
        action_type = action.get("type")
        score = 0.0
        idx = 0  # デフォルト値を設定
        type_weight = individual[idx] if individual and len(individual) > 0 else 0.5
        
        # 直前の移動先を記憶
        if not hasattr(self, 'previous_cities'):
            self.previous_cities = {}
        
        # 治療薬発見の優先度を最大に
        if action_type == "discover_cure":
            # 条件チェック（研究所にいるか、カードが十分あるか）
            if player.city.has_research_station and action.get("cards") and len(action.get("cards")) >= 4:
                return 20.0  # 他のどのアクションよりも優先

        if action_type == "move":
            target_city = action.get("target_city")
            
            # 同じ都市への移動、または1ターン以内に訪れた都市への移動は大幅減点
            if target_city == player.city:
                return -20.0
            
            # 前回移動した都市への再移動を防止
            player_id = player.id if hasattr(player, 'id') else id(player)
            if player_id in self.previous_cities and self.previous_cities[player_id] == target_city.name:
                return -10.0
            
            # 以降は既存のスコアリング...
            if len(individual) > 0:
                idx = 0 % len(individual)
                type_weight = individual[idx]
                
                if target_city and target_city != player.city:  # プレイヤーが現在いる都市と異なる場合のみスコア
                    if idx + 1 < len(individual):
                        infection_weight = individual[idx + 1]
                        score += infection_weight * target_city.infection_level
                    
                    if idx + 2 < len(individual):
                        station_weight = individual[idx + 2]
                        if target_city.has_research_station:
                            score += station_weight * 2.0
                else:
                    # 同じ都市への移動は非常に低いスコア
                    score = -10.0
        elif action_type == "treat":
            city = action.get("city", player.city)
            # 感染レベルが3の都市を優先的に治療（アウトブレイク防止）
            if city and city.infection_level >= 3:
                return 12.0  # 建設より低く、移動より高く
            score += type_weight * 1.5
        elif action_type == "build":
            score += type_weight * 2.5  # 研究所建設の優先度を上げる
        elif action_type == "share_knowledge":
            score += type_weight * 2.0  # 知識共有も重要
        
        if action_type == "move":
            target_city = action.get("target_city")
            if target_city:
                if idx + 1 < len(individual):
                    infection_weight = individual[idx + 1]
                    score += infection_weight * target_city.infection_level
                
                if idx + 2 < len(individual):
                    station_weight = individual[idx + 2]
                    if target_city.has_research_station:
                        score += station_weight * 2.0
        
        elif action_type == "treat":
            target_city = action.get("city") or player.city
            if target_city:
                if idx + 3 < len(individual):
                    treatment_weight = individual[idx + 3]
                    score += treatment_weight * target_city.infection_level * 0.8
                
                # for preventing outbreaks
                if target_city.infection_level == 3 and idx + 4 < len(individual):
                    outbreak_weight = individual[idx + 4]
                    score += outbreak_weight * 3.0
        
        elif action_type == "build":
            target_city = action.get("city") or player.city
            if target_city:
                if idx + 5 < len(individual):
                    centrality_weight = individual[idx + 5]
                    score += centrality_weight * len(target_city.neighbours) * 0.4
                
                station_cities = [c for c in simulation.cities if c.has_research_station]
                if station_cities and idx + 6 < len(individual):
                    distance_weight = individual[idx + 6]
                    max_distance = 0
                    for station_city in station_cities:
                        distance = 1 if station_city in target_city.neighbours else 2
                        max_distance = max(max_distance, distance)
                    
                    score += distance_weight * max_distance * 0.5
            
            # 既存の研究所数をチェック
            research_stations = [c for c in simulation.cities if c.has_research_station]
            if len(research_stations) < 5:  # 研究所は5つまで
                return 15.0  # 治療薬発見よりは低いが他より高い
        
        if hasattr(player, 'role') and player.role:
            role_name = player.role.name
            role_idx = {"Medic": 7, "Scientist": 8, "Researcher": 9, "Operations Expert": 10}.get(role_name, 11)
            
            if idx + role_idx < len(individual):
                role_weight = individual[idx + role_idx]

                if role_name == "Medic" and action_type == "treat":
                    score += role_weight * 2.0  # enhanced treatment
                elif role_name == "Scientist" and action_type == "build":
                    score += role_weight * 1.5  # enhanced research station building
                elif role_name == "Operations Expert" and action_type == "build":
                    score += role_weight * 2.5  # enhanced building
        
        if action_type in self.success_counter:
            success_rate = self.success_counter[action_type].get("success", 0) / max(1, self.success_counter[action_type].get("total", 1))
            score *= (0.5 + 0.5 * success_rate) 
        
        return score
    
    def decide_action(self, player, simulation):
        # 手札情報のデバッグ出力
        print(f"DEBUG: Player {player.name} has {len(player.hand)} cards:")
        for i, card in enumerate(player.hand):
            color = getattr(card, 'color', 'unknown')
            city_name = getattr(card, 'city_name', getattr(getattr(card, 'city', None), 'name', 'unknown'))
            print(f"  Card {i+1}: {color} - {city_name}")
        
        # 研究所の位置を出力
        research_cities = [city.name for city in simulation.cities if city.has_research_station]
        print(f"DEBUG: Research stations at: {', '.join(research_cities)}")
        
        # 修正: 治療薬発見のための準備が整っている場合は最優先
        cards_by_color = {}
        for card in player.hand:
            if hasattr(card, 'color') and card.color and card.color != "INF":
                color = card.color
                if color not in cards_by_color:
                    cards_by_color[color] = []
                cards_by_color[color].append(card)
        
        required_cards = 4 if (hasattr(player, 'role') and player.role.name == "Scientist") else 5
        
        # 治療薬発見可能なら研究所に急ぐ
        for color, cards in cards_by_color.items():
            if len(cards) >= required_cards:
                # 研究所がある都市を探す
                research_cities = [city for city in simulation.cities if city.has_research_station]
                
                if research_cities:
                    # 手持ちの都市カードで直接移動できる研究所のある都市があれば最優先
                    for city in research_cities:
                        for card in player.hand:
                            if hasattr(card, 'city_name') and card.city_name == city.name:
                                return {"type": "move", "target_city": city, "method": "direct_flight", "card": card}
        
        if not self.best_population:
            self.toolbox.register("evaluate", self.evaluate_fitness, 
                                 player=player, simulation=simulation)
            
            pop = self.toolbox.population(n=self.population_size)
            self.best_population = pop
        
        if random.random() < 0.2 or not self.best_individual:  # 20% probability or initial call
            self.toolbox.register("evaluate", self.evaluate_fitness, 
                                player=player, simulation=simulation)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("max", np.max)
            
            pop, logbook = algorithms.eaSimple(
                self.best_population, self.toolbox, 
                cxpb=self.crossover_rate, mutpb=self.mutation_rate, 
                ngen=self.generations, stats=stats, verbose=False)
            
            self.best_population = pop
            self.best_individual = tools.selBest(pop, 1)[0]
            self.generation_count += 1
            

            if hasattr(logbook, 'select') and 'max' in logbook.select('max'):
                best_fitness = logbook.select('max')[-1]
                self.fitness_history.append(best_fitness)
        
        possible_actions = self.get_possible_actions(player, simulation)  # BaseAgentから継承したメソッドを使用
        
        if not possible_actions:
            return None
            
        action_scores = []
        for action in possible_actions:
            score = self._score_action(action, self.best_individual, player, simulation)
            action_scores.append((action, score))
            
        best_action = max(action_scores, key=lambda x: x[1])[0]
        
        # 移動アクションの場合は移動先を記録
        if best_action.get("type") == "move":
            player_id = player.id if hasattr(player, 'id') else id(player)
            target_city = best_action.get("target_city")
            if target_city:
                if not hasattr(self, 'previous_cities'):
                    self.previous_cities = {}
                self.previous_cities[player_id] = target_city.name
        
        best_score = max(action_scores, key=lambda x: x[1])[1]
        print(f"DEBUG: EAAgent selecting {best_action.get('type')} action with score {best_score}")
        
        action_type = best_action.get("type", "unknown")
        if action_type not in self.success_counter:
            self.success_counter[action_type] = {"total": 0, "success": 0}
        
        self.success_counter[action_type]["total"] += 1
        
        self.record_action("ea_decision", {
            "action": best_action,
            "score": max(action_scores, key=lambda x: x[1])[1],
            "generation": self.generation_count
        })
        
        return best_action
    
    def update_success_counter(self, action_type, success):
        if action_type in self.success_counter:
            if success:
                self.success_counter[action_type]["success"] += 1
    
    def save_state(self, filename="ea_agent_state.pkl"):
        save_data = {
            "best_population": self.best_population,
            "best_individual": self.best_individual,
            "generation_count": self.generation_count,
            "fitness_history": self.fitness_history,
            "success_counter": self.success_counter,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate
        }
        
        with open(filename, "wb") as f:
            pickle.dump(save_data, f)
        print(f" EA agent state saved in {filename}")

    def load_state(self, filename="ea_agent_state.pkl"):
        try:
            with open(filename, "rb") as f:
                save_data = pickle.load(f)
            
            self.best_population = save_data.get("best_population", [])
            self.best_individual = save_data.get("best_individual", None)
            self.generation_count = save_data.get("generation_count", 0)
            self.fitness_history = save_data.get("fitness_history", [])
            self.success_counter = save_data.get("success_counter", {})
            self.crossover_rate = save_data.get("crossover_rate", self.crossover_rate)
            self.mutation_rate = save_data.get("mutation_rate", self.mutation_rate)
            
            print(f"Loaded  EA agent state from {filename}")
            return True
        except Exception as e:
            print(f"Error loading EA state: {e}")
            return False

_global_ea_agent = None

def ea_agent_strategy(player):
    global _global_ea_agent
    
    import os
    agent_state_dir = "./agents_state"
    os.makedirs(agent_state_dir, exist_ok=True)
    state_file = os.path.join(agent_state_dir, "ea_agent_state.pkl")
    
    if _global_ea_agent is None:
        _global_ea_agent = EAAgent()
        print(f"Created new EA agent (state file: {state_file})")
        _global_ea_agent.load_state(filename=state_file)
    
    action = _global_ea_agent.decide_action(player, player.simulation)
    
    if not action:
        return None
    
    # アクション処理の前にデバッグ出力を追加
    print(f"DEBUG: EA Strategy processing action: {action.get('type')}")
    
    # 各アクション種別に対応する処理
    if action.get("type") == "move":
        target_city = action.get("target_city")
        method = action.get("method", "standard")
        card = action.get("card")
        
        if target_city and target_city != player.city:
            return {
                "type": "move", 
                "target_city": target_city,
                "method": method,
                "card": card
            }
        return {"type": "pass"}
    
    elif action.get("type") == "treat":
        target_city = action.get("city") or player.city
        color = action.get("color", "Blue")
        if target_city.infection_level > 0:
            return {
                "type": "treat", 
                "city": target_city,
                "color": color
            }
        return {"type": "pass"}
    
    elif action.get("type") == "build":
        # 研究所建設は非常に重要なアクション
        city_card = None
        for card in player.hand:
            if (hasattr(card, 'city_name') and card.city_name == player.city.name) or \
               (hasattr(card, 'city') and hasattr(card.city, 'name') and card.city.name == player.city.name):
                city_card = card
                break
                
        return {
            "type": "build", 
            "city": player.city,
            "card": city_card  # カードも指定する必要がある場合
        }
    
    elif action.get("type") == "discover_cure":
        # 治療薬発見は勝利条件に直結する最重要アクション
        color = action.get("color")
        cards = action.get("cards", [])
        if color and cards and len(cards) >= 4:
            print(f"DEBUG: Attempting to discover cure for {color} with {len(cards)} cards")
            return {
                "type": "discover_cure",
                "color": color,
                "cards": cards
            }
        return {"type": "pass"}
    
    elif action.get("type") == "share_knowledge":
        # 知識共有も実装する
        direction = action.get("direction")
        target_player = action.get("target_player")
        card = action.get("card")
        
        if direction and target_player and card:
            return {
                "type": "share_knowledge",
                "direction": direction,
                "target_player": target_player,
                "card": card
            }
    
    elif action.get("type") == "pass":
        return {"type": "pass"}
    
    # どのアクションにも当てはまらない場合
    print("DEBUG: EA Strategy fell through to default action")
    return {"type": "pass"}