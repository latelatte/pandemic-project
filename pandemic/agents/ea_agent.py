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

        total_infection = sum(city.infection_level for city in simulation.cities)
        max_infection = 3 * len(simulation.cities)
        
        infection_control = 1.0 - (total_infection / max_infection)
        
        total_stations = sum(1 for city in simulation.cities if city.has_research_station)
        station_coverage = total_stations / len(simulation.cities)
        
        station_network_score = 0.0
        cities_with_stations = [city for city in simulation.cities if city.has_research_station]
        
        for city in cities_with_stations:
            adjacent_stations = sum(1 for neighbor in city.neighbours if neighbor.has_research_station)
            if adjacent_stations == 0:
                station_network_score += 0.2
                
        cure_progress = 0.0
        if hasattr(simulation, 'discovered_cures'):
            cure_progress = len(simulation.discovered_cures) / 4.0 
        
        outbreak_safety = 1.0 - (simulation.outbreak_count / simulation.outbreak_limit)

        role_bonus = 0.0
        if hasattr(player, 'role') and player.role:
            role_name = player.role.name
            if role_name == "Medic" and total_infection < max_infection * 0.5:
                role_bonus = 0.2 
            elif role_name == "Scientist" and cure_progress > 0:
                role_bonus = 0.2 
            elif role_name == "Operations Expert" and station_coverage > 0.3:
                role_bonus = 0.2 
 
        fitness = (
            0.35 * infection_control + 
            0.20 * station_coverage +   
            0.10 * station_network_score + 
            0.25 * cure_progress +  
            0.05 * outbreak_safety +   
            0.05 * role_bonus           
        )
        
        # 0-1の範囲に正規化
        return (max(0, min(1, fitness)),)
    
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
        if not player.city.has_research_station:
            for card in player.hand:
                if hasattr(card, 'city_name') and card.city_name == player.city.name:
                    actions.append({
                        "type": "build",
                        "city": player.city
                    })
                    break
        
        return actions
    
    def _score_action(self, action, individual, player, simulation):
        action_type = action.get("type")
        score = 0.0
        
        if len(individual) > 0:
            idx = 0 % len(individual)
            type_weight = individual[idx]
            
            if action_type == "move":
                score += type_weight * 1.0
            elif action_type == "treat":
                score += type_weight * 1.5
            elif action_type == "build":
                score += type_weight * 1.3
        
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
        """進化的アルゴリズムを使用して最適な行動を決定"""
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
        
        possible_actions = self._get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
            
        action_scores = []
        for action in possible_actions:
            score = self._score_action(action, self.best_individual, player, simulation)
            action_scores.append((action, score))
            
        best_action = max(action_scores, key=lambda x: x[1])[0]
        
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
        print(f"Created new  EA agent (state file: {state_file})")
        _global_ea_agent.load_state(filename=state_file)
    
    action = _global_ea_agent.decide_action(player, player.simulation)
    
    if random.random() < 0.01:
        _global_ea_agent.save_state(filename=state_file)
    
    if not action:
        return None
    
    
    if action.get("type") == "move":
        target_city = action.get("target_city")
        if target_city:
            return {"type": "move", "target": target_city}
    
    elif action.get("type") == "treat":
        target_city = action.get("city") or player.city
        if target_city.infection_level > 0:
            return {"type": "treat", "target": target_city}
    
    elif action.get("type") == "build":
        return {"type": "build", "target": player.city}
    
    return None