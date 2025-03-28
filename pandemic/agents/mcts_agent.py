import math
import random
import time
from copy import deepcopy
import numpy as np
import pickle
import os
from pandemic.agents.baseline_agents import BaseAgent

class MCTSNode:
    def __init__(self, state, parent=None, action=None, exploration_weight=1.4):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self._get_untried_actions()
        self.exploration_weight = exploration_weight
        
    def _get_untried_actions(self):
        actions = []
        player = self.state.get("current_player")
        
        if player and player.city:
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
            
            # build research station
            if not player.city.has_research_station:
                for card in player.hand:
                    if hasattr(card, 'city_name') and card.city_name == player.city.name:
                        actions.append({
                            "type": "build",
                            "city": player.city
                        })
                        break
        
        return actions
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def is_terminal(self):

        return self.state.get("game_over", False)
    
    def select_child(self):
        # UCB1 = exploitation + exploration
        # exploitation = avg_value / (child visits + 1e-10)
        # exploration = sqrt(2 * ln(parent visits) / child visits)
        
        log_total = math.log(self.visits + 1e-10)
        
        def ucb_score(child):
            exploitation = child.value / (child.visits + 1e-10)
            exploration = self.exploration_weight * math.sqrt(log_total / (child.visits + 1e-10))
            return exploitation + exploration
        
        return max(self.children, key=ucb_score)
    
    def expand(self):
        if not self.untried_actions:
            return None
            
        action = self.untried_actions.pop()
        next_state = self.apply_action(self.state, action)
        child = MCTSNode(next_state, parent=self, action=action, exploration_weight=self.exploration_weight)
        self.children.append(child)
        return child
        
    def apply_action(self, state, action):
        new_state = deepcopy(state)
        player = new_state.get("current_player")
        
        if not player:
            return new_state
        
        action_type = action.get("type")
        
        if action_type == "move":
            target_city = action.get("target_city")
            if target_city:
                for city in new_state.get("cities", []):
                    if city.name == target_city.name:
                        player.city = city
                        break
        
        elif action_type == "treat":
            city = action.get("city") or player.city
            if city and city.infection_level > 0:
                city.infection_level -= 1
                
                for disease in new_state.get("diseases", []):
                    if hasattr(disease, 'cured') and disease.cured:
                        city.infection_level = 0
                        break
        
        elif action_type == "build":
            if player.city:
                player.city.has_research_station = True
        
        game_over = False
        win = False
        
        if "diseases" in new_state:
            all_cured = True
            for disease in new_state["diseases"]:
                if hasattr(disease, 'cured') and not disease.cured:
                    all_cured = False
                    break
            
            if all_cured:
                game_over = True
                win = True
        
        if "outbreak_count" in new_state and "outbreak_limit" in new_state:
            if new_state["outbreak_count"] >= new_state["outbreak_limit"]:
                game_over = True
                win = False
        
        new_state["game_over"] = game_over
        new_state["win"] = win
        
        return new_state
        
    def update(self, result):
        self.visits += 1
        self.value += result
    
    def rollout_policy(self, state):
        """Rollout policy (heuristic)"""
        player = state.get("current_player")
        
        if not player or not player.city:
            return None
            
        possible_actions = []
        
        # move
        for neighbor in player.city.neighbours:
            possible_actions.append({
                "type": "move", 
                "target_city": neighbor
            })
        
        # treat action
        if player.city.infection_level > 0:
            for _ in range(player.city.infection_level):
                possible_actions.append({
                    "type": "treat",
                    "city": player.city
                })
        
        # research station
        if not player.city.has_research_station:
            for card in player.hand:
                if hasattr(card, 'city_name') and card.city_name == player.city.name:
                    for _ in range(3):
                        possible_actions.append({
                            "type": "build",
                            "city": player.city
                        })
                    break
        
        if not possible_actions:
            return None
            
        return random.choice(possible_actions)

class MCTSAgent(BaseAgent):
    def __init__(self, name="-MCTS", simulation_count=100, exploration_weight=1.4,
                max_rollout_depth=10, time_limit=0.95, progressive_widening=True):
        super().__init__(name)
        self.simulation_count = simulation_count
        self.exploration_weight = exploration_weight
        self.max_rollout_depth = max_rollout_depth  # limit for depth of simulation
        self.max_time = time_limit  # maximum time limit for decision making
        self.progressive_widening = progressive_widening  # using progressive widening
 
        self.node_cache = {}

        self.action_stats = {}
        self.visit_counts = {}
        self.value_sum = {}
        
        self.last_simulation_count = 0
        self.thinking_time = 0
        
    def _extract_state(self, simulation, player):
        state = {
            "current_player": player,
            "cities": simulation.cities,
            "players": simulation.players,
            "diseases": getattr(simulation, "diseases", []),
            "discovered_cures": getattr(simulation, "discovered_cures", []),
            "outbreak_count": simulation.outbreak_count,
            "outbreak_limit": simulation.outbreak_limit,
            "game_over": simulation.game_over,
            "win": False
        }
        return state
    
    def _state_hash(self, state):
        """calculate a hash for the state"""
        player = state.get("current_player")
        
        components = []
        
        if player and player.city:
            components.append(f"player_pos:{player.city.name}")

        infection_str = ""
        for city in state.get("cities", []):
            if city.infection_level > 0:
                infection_str += f"{city.name}:{city.infection_level},"
        components.append(f"infection:{infection_str}")
        
        stations_str = ""
        for city in state.get("cities", []):
            if city.has_research_station:
                stations_str += f"{city.name},"
        components.append(f"stations:{stations_str}")
        
        components.append(f"outbreaks:{state.get('outbreak_count', 0)}")
        
        cures_str = ""
        for cure in state.get("discovered_cures", []):
            cures_str += f"{cure},"
        components.append(f"cures:{cures_str}")
        
        return hash(tuple(components))
    
    def _evaluate_state(self, state):
        """heuristic evaluation of the state"""

        if state.get("game_over", False):
            return 1.0 if state.get("win", False) else 0.0
        
        score = 0.5  # base score
        
        # infection level
        total_infection = 0
        max_possible_infection = 0
        
        for city in state.get("cities", []):
            total_infection += city.infection_level
            max_possible_infection += 3  # assuming max infection level is 3
        
        # infection control score
        if max_possible_infection > 0:
            infection_control_score = 1.0 - (total_infection / max_possible_infection)
            score += 0.2 * infection_control_score
        
        # num of research stations
        research_stations = sum(1 for city in state.get("cities", []) if city.has_research_station)
        station_score = min(1.0, research_stations / 4.0)  # 4つあれば十分と仮定
        score += 0.1 * station_score
        
        # cures discovered
        cures_discovered = len(state.get("discovered_cures", []))
        cure_score = cures_discovered / 4.0  # 4種類の治療薬を想定
        score += 0.3 * cure_score
        
        # outbreaks
        outbreak_count = state.get("outbreak_count", 0)
        outbreak_limit = state.get("outbreak_limit", 8)
        outbreak_safety = 1.0 - (outbreak_count / outbreak_limit)
        score += 0.1 * outbreak_safety
        
        return score
    
    def _simulate(self, node, depth=0):
        """run rollout simulation"""
        if node.is_terminal():
            return 1.0 if node.state.get("win", False) else 0.0
            
        if depth >= self.max_rollout_depth:
            return self._evaluate_state(node.state)
            
        action = node.rollout_policy(node.state)
        if not action:
            return self._evaluate_state(node.state)
            
        next_state = node.apply_action(node.state, action)
        
        # 再帰的にシミュレーション続行
        node = MCTSNode(next_state, exploration_weight=self.exploration_weight)
        return self._simulate(node, depth + 1)
    
    def decide_action(self, player, simulation):
        start_time = time.time()

        root_state = self._extract_state(simulation, player)
        state_hash = self._state_hash(root_state)

        if state_hash in self.node_cache:
            root = self.node_cache[state_hash]
        else:
            root = MCTSNode(root_state, exploration_weight=self.exploration_weight)
        
        simulation_count = 0
        
        while (time.time() - start_time < self.max_time and 
               simulation_count < self.simulation_count):
               
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child()

            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
                
            result = self._simulate(node)

            while node is not None:
                node.update(result)
                node = node.parent
                
            simulation_count += 1
        
        if not root.children:
            return None
            
        best_child = max(root.children, key=lambda c: c.visits)
        best_action = best_child.action
        
        if len(self.node_cache) > 1000:  # limit cache size
            self.node_cache = {}
        self.node_cache[state_hash] = root
        
        action_key = self._action_to_key(best_action)
        state_key = str(state_hash)
        
        if state_key not in self.action_stats:
            self.action_stats[state_key] = {}
            self.visit_counts[state_key] = {}
            self.value_sum[state_key] = {}
            
        if action_key not in self.action_stats[state_key]:
            self.action_stats[state_key][action_key] = 0
            self.visit_counts[state_key][action_key] = 0
            self.value_sum[state_key][action_key] = 0.0
 
        child_visits = best_child.visits
        child_value = best_child.value
        
        self.visit_counts[state_key][action_key] += child_visits
        self.value_sum[state_key][action_key] += child_value
        
        if self.visit_counts[state_key][action_key] > 0:
            self.action_stats[state_key][action_key] = (
                self.value_sum[state_key][action_key] / self.visit_counts[state_key][action_key]
            )

        self.last_simulation_count = simulation_count
        self.thinking_time = time.time() - start_time
        
        self.record_action("mcts_decision", {
            "action": best_action,
            "simulations": simulation_count,
            "time": self.thinking_time,
            "avg_value": best_child.value / max(1, best_child.visits)
        })
        
        return best_action
    
    def _action_to_key(self, action):
        """convert action to a string key for statistics"""
        if not action:
            return "None"
        
        action_type = action.get("type", "unknown")
        
        if action_type == "move":
            target_city = action.get("target_city")
            city_name = target_city.name if target_city else "unknown"
            return f"move-{city_name}"
            
        elif action_type == "treat":
            city = action.get("city")
            city_name = city.name if city else "current"
            return f"treat-{city_name}"
            
        elif action_type == "build":
            return "build"
            
        return str(action)
    
    def save_state(self, filepath="mcts_agent_state.pkl"):
        # ignore node cache since it is too large
        save_data = {
            "action_stats": self.action_stats,
            "visit_counts": self.visit_counts,
            "value_sum": self.value_sum,
            "exploration_weight": self.exploration_weight,
            "last_simulation_count": self.last_simulation_count,
            "thinking_time": self.thinking_time
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        print(f" MCTS agent state saved in {filepath}")
        
    def load_state(self, filepath="mcts_agent_state.pkl"):
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    save_data = pickle.load(f)
                    
                self.action_stats = save_data.get("action_stats", {})
                self.visit_counts = save_data.get("visit_counts", {})
                self.value_sum = save_data.get("value_sum", {})
                self.exploration_weight = save_data.get("exploration_weight", self.exploration_weight)
                self.last_simulation_count = save_data.get("last_simulation_count", 0)
                self.thinking_time = save_data.get("thinking_time", 0)
                
                print(f"Loaded  MCTS agent state from {filepath}")
                return True
            
            return False
        except Exception as e:
            print(f"Error loading MCTS state: {e}")
            return False

_global_mcts_agent = None

def mcts_agent_strategy(player):
    global _global_mcts_agent
    
    import os
    agent_state_dir = "./agents_state"
    os.makedirs(agent_state_dir, exist_ok=True)
    state_file = os.path.join(agent_state_dir, "mcts_agent_state.pkl")
    
    if _global_mcts_agent is None:
        _global_mcts_agent = MCTSAgent()
        print(f"Created new  MCTS agent (state file: {state_file})")
        _global_mcts_agent.load_state(filepath=state_file)
    
    action = _global_mcts_agent.decide_action(player, player.simulation)
    
    if random.random() < 0.01:
        _global_mcts_agent.save_state(filepath=state_file)
    
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