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

# ダミー戦略関数（SimulationRunnerから呼び出されるため）
def random_agent_strategy(player):
    agent = RandomAgent()
    while player.actions_remaining > 0:
        action = agent.decide_action(player, player.simulation)
        if not action:
            break  # 行動不可能
            
        # アクション実行（タイプに応じた処理）
        if action["type"] == "move":
            player.move_to(action["target_city"])
        elif action["type"] == "treat":
            action["city"].treat_infection()
        # その他のアクション処理
        
        player.actions_remaining -= 1