import random

class BaseAgent:
    """すべてのエージェントの基底クラス"""
    def __init__(self, name):
        self.name = name
        self.actions = []  # 記録用

    def decide_action(self, player, simulation):
        """アクションを決定するメソッド（サブクラスで実装）"""
        raise NotImplementedError
        
    def record_action(self, action_type, details):
        """アクションを記録"""
        self.actions.append({
            "type": action_type,
            "details": details
        })

class RandomAgent(BaseAgent):
    """ランダムに行動を選択するエージェント（ベースライン）"""
    def __init__(self, name="Random"):
        super().__init__(name)
    
    def decide_action(self, player, simulation):
        """利用可能なアクションからランダムに選択"""
        possible_actions = self._get_possible_actions(player, simulation)
        if not possible_actions:
            return None  # 行動不可
        
        chosen_action = random.choice(possible_actions)
        self.record_action(chosen_action["type"], chosen_action)
        return chosen_action
        
    def _get_possible_actions(self, player, simulation):
        """可能なアクションリストを取得"""
        actions = []
        
        # 移動アクション
        for neighbor in player.city.neighbours:
            actions.append({
                "type": "move",
                "target_city": neighbor
            })
            
        # 治療アクション（現在の都市に感染がある場合）
        if player.city.infection_level > 0:
            actions.append({
                "type": "treat",
                "city": player.city
            })
            
        # その他のアクションも追加（研究所建設、カード交換など）
        
        return actions

# RandomAgentの改良例
def random_agent_strategy(player):
    """ランダム行動選択の戦略"""
    # 移動するか治療するかをランダム決定
    action_type = random.choice(["move", "treat"])
    
    if action_type == "move" and player.city and player.city.neighbours:
        # 隣接都市へランダム移動
        target = random.choice(player.city.neighbours)
        return {"type": "move", "target": target}
    elif action_type == "treat" and player.city and player.city.infection_level > 0:
        # 現在地の治療
        return {"type": "treat", "target": player.city}
    
    # 他に選択肢がない場合、ランダムな隣接都市へ移動
    if player.city and player.city.neighbours:
        target = random.choice(player.city.neighbours)
        return {"type": "move", "target": target}
    
    return None  # どのアクションも不可能な場合