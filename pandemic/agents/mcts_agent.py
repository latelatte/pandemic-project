import math
import random
import time
from copy import deepcopy
from pandemic.agents.baseline_agents import BaseAgent
import pickle

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state 
        self.parent = parent 
        self.action = action 
        self.children = [] 
        self.visits = 0 
        self.untried_actions = self._get_untried_actions()
        
    def _get_untried_actions(self):
        actions = []
        player = self.state.get("current_player")
        if player and player.city:
            for neighbor in player.city.neighbours:
                actions.append({
                    "type": "move", 
                    "target_city": neighbor
                })
            if player.city.infection_level > 0:
                actions.append({
                    "type": "treat",
                    "city": player.city
                })
        return actions
        
    def select_child(self, exploration_weight=1.0):
        # UCB1 = value + exploration_weight * sqrt(2 * ln(parent_visits) / child_visits)
        return max(self.children, key=lambda c: 
            c.value / c.visits + exploration_weight * math.sqrt(2 * math.log(self.visits) / c.visits))
    
    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.apply_action(self.state, action)
        child = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child
        
    def apply_action(self, state, action):
        new_state = deepcopy(state)

        return new_state
        
    def update(self, result):
        self.visits += 1
        self.value += result 

class MCTSAgent(BaseAgent):
    def __init__(self, name="MCTS", simulation_count=100, exploration_weight=1.0):
        super().__init__(name)
        self.simulation_count = simulation_count
        self.exploration_weight = exploration_weight
        self.max_time = 1.0  # max time per action in seconds 
        
    def decide_action(self, player, simulation):
        start_time = time.time()
        root = MCTSNode(self._extract_state(simulation, player))
        
        simulation_count = 0
        while (time.time() - start_time < self.max_time and 
               simulation_count < self.simulation_count):
            node = root
            while node.untried_actions == [] and node.children != []:
                node = node.select_child(self.exploration_weight)
                
            if node.untried_actions != []:
                node = node.expand()
                
            result = self._simulate(node.state)
            
            while node is not None:
                node.update(result)
                node = node.parent
                
            simulation_count += 1
            
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        best_action = best_child.action if best_child else None
        
        self.record_action("mcts_decision", {
            "action": best_action,
            "simulations": simulation_count,
            "time": time.time() - start_time
        })
        
        return best_action
    
    def _extract_state(self, simulation, player):
        state = {
            "cities": deepcopy(simulation.cities),
            "players": deepcopy(simulation.players),
            "current_player": player,

        }
        return state
    
    def _simulate(self, state):

        # 本来はゲームロジックに沿った実装が必要

        return random.random() < 0.5

    def save_state(self, filepath="mcts_agent_state.pkl"):
        """エージェントの状態を保存"""
        save_data = {
            "action_stats": getattr(self, "action_stats", {}),
            "visit_counts": getattr(self, "visit_counts", {}),
            "value_sum": getattr(self, "value_sum", {}),
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        print(f"MCTS agent state saved in {filepath}")
        
    def load_state(self, filepath="mcts_agent_state.pkl"):

        try:
            with open(filepath, "rb") as f:
                save_data = pickle.load(f)
                
            self.action_stats = save_data.get("action_stats", {})
            self.visit_counts = save_data.get("visit_counts", {})
            self.value_sum = save_data.get("value_sum", {})
            print(f"loaded MCTS agent state from {filepath} .")
            return True
        except:
            print(f"{filepath} not found or invalid.")
            return False
            
    def update_action_stats(self, state_hash, action, result):
        if not hasattr(self, "action_stats"):
            self.action_stats = {}
        if not hasattr(self, "visit_counts"):
            self.visit_counts = {}
        if not hasattr(self, "value_sum"):
            self.value_sum = {}
            
        if state_hash not in self.action_stats:
            self.action_stats[state_hash] = {}
            self.visit_counts[state_hash] = {}
            self.value_sum[state_hash] = {}
            
        action_key = self._action_to_key(action)
        
        if action_key not in self.action_stats[state_hash]:
            self.action_stats[state_hash][action_key] = 0
            self.visit_counts[state_hash][action_key] = 0
            self.value_sum[state_hash][action_key] = 0.0
            
        self.visit_counts[state_hash][action_key] += 1
        self.value_sum[state_hash][action_key] += result
        self.action_stats[state_hash][action_key] = self.value_sum[state_hash][action_key] / self.visit_counts[state_hash][action_key]
    
    def _action_to_key(self, action):
        """convert action to a string key"""
        if not action:
            return "None"
        
        if action.get("type") == "move":
            city_name = action.get("target_city").name if action.get("target_city") else "unknown"
            return f"move-{city_name}"
        elif action.get("type") == "treat":
            city_name = action.get("city").name if action.get("city") else "current"
            return f"treat-{city_name}"
        
        return str(action)

_global_mcts_agent = None

def mcts_agent_strategy(player):
    global _global_mcts_agent
    
    import os
    log_dir = player.simulation.log_dir if hasattr(player.simulation, 'log_dir') else "./logs"
    state_file = os.path.join(log_dir, "mcts_agent_state.pkl")
    
    
    if _global_mcts_agent is None:
        _global_mcts_agent = MCTSAgent()
        print(f"creating new MCTS agent（saved in: {state_file}）")
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
    

    return None