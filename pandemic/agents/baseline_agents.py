import random

class BaseAgent:
    """base class for all agents"""
    def __init__(self, name):
        self.name = name
        self.actions = []  # 記録用

    def decide_action(self, player, simulation):
        raise NotImplementedError
        
    def record_action(self, action_type, details):
        self.actions.append({
            "type": action_type,
            "details": details
        })

class RandomAgent(BaseAgent):
    """random agent"""
    def __init__(self, name="Random"):
        super().__init__(name)
    
    def decide_action(self, player, simulation):
        possible_actions = self._get_possible_actions(player, simulation)
        if not possible_actions:
            return None  # 行動不可
        
        chosen_action = random.choice(possible_actions)
        self.record_action(chosen_action["type"], chosen_action)
        return chosen_action
        
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
            
        # TODO: add other actions like build_research_station, share_knowledge, etc.
        
        return actions


def random_agent_strategy(player):
    """ランダム行動選択の戦略"""

    action_type = random.choice(["move", "treat"])
    
    if action_type == "move" and player.city and player.city.neighbours:
        target = random.choice(player.city.neighbours)
        return {"type": "move", "target": target}
    elif action_type == "treat" and player.city and player.city.infection_level > 0:
        return {"type": "treat", "target": player.city}
    

    if player.city and player.city.neighbours:
        target = random.choice(player.city.neighbours)
        return {"type": "move", "target": target}
    
    return None