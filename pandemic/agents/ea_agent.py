import random
from deap import base, creator, tools, algorithms
import numpy as np
from pandemic.agents.baseline_agents import BaseAgent

class Individual:
    def __init__(self, genome_length=20, genome=None):
        if genome is None:
            self.genome = np.random.random(genome_length)
        else:
            self.genome = genome
        self.fitness = 0.0
        
    def mutate(self, mutation_rate=0.1):
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] = random.random()
        return self
    
    def crossover(self, other, crossover_rate=0.7):
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
    def __init__(self, name="DEAP-EA", population_size=50, generations=10):
        super().__init__(name)
        self.population_size = population_size
        self.generations = generations
        self.genome_length = 20
        
        self._setup_deap()
        
    def _setup_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_float, self.genome_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def evaluate_fitness(self, individual, player, simulation):
        
        total_infection = sum(city.infection_level for city in simulation.cities)
        total_stations = sum(1 for city in simulation.cities if city.has_research_station)
        
        fitness = 1.0 - (total_infection / (len(simulation.cities) * 5))
        fitness += 0.2 * (total_stations / len(simulation.cities))
        
        return (max(0, min(1, fitness)),)
    
    def decide_action(self, player, simulation):
        self.toolbox.register("evaluate", self.evaluate_fitness, 
                             player=player, simulation=simulation)
        
        pop = self.toolbox.population(n=self.population_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox, cxpb=0.7, mutpb=0.2, 
            ngen=self.generations, stats=stats, verbose=False)
        
        self.best_population = pop
        best_ind = tools.selBest(pop, 1)[0]

        possible_actions = self._get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
            
        action_scores = []
        for i, action in enumerate(possible_actions):
            idx = i % len(best_ind)
            score = best_ind[idx]
            
            if action["type"] == "treat" and player.city.infection_level > 0:
                score *= 1.2 
                
            action_scores.append((action, score))
            
        best_action = max(action_scores, key=lambda x: x[1])[0]
        
        self.record_action("ea_decision", {
            "action": best_action,
            "fitness": best_ind.fitness.values[0]
        })
        
        return best_action

    def save_state(self, filename="ea_agent_state.pkl"):
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(self.best_population, f)
        print(f"state saved in {filename}.")

    def load_state(self, filename="ea_agent_state.pkl"):
        import pickle
        try:
            with open(filename, "rb") as f:
                self.best_population = pickle.load(f)
            print(f"loaded state from {filename} .")
            return True
        except:
            print(f"{filename} not found or invalid.")
            return False

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

_global_ea_agent = None

def ea_agent_strategy(player):
    global _global_ea_agent
    
    import os
    log_dir = player.simulation.log_dir if hasattr(player.simulation, 'log_dir') else "./logs"
    state_file = os.path.join(log_dir, "ea_agent_state.pkl")
    
    if _global_ea_agent is None:
        _global_ea_agent = EAAgent()
        print(f"created new EA agent（saved in: {state_file}）")
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
    
    
    return None

