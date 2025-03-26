import math
import random
import time
from copy import deepcopy
from pandemic.agents.baseline_agents import BaseAgent
import pickle

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # ゲーム状態
        self.parent = parent  # 親ノード
        self.action = action  # このノードに至ったアクション
        self.children = []  # 子ノード
        self.visits = 0  # 訪問回数
        self.value = 0.0  # 価値（勝率）
        self.untried_actions = self._get_untried_actions()  # 未試行のアクション
        
    def _get_untried_actions(self):
        """未試行のアクション一覧を取得"""
        # 実装を追加
        actions = []
        player = self.state.get("current_player")
        if player and player.city:
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
        return actions
        
    def select_child(self, exploration_weight=1.0):
        """UCB1に基づく子ノード選択"""
        # UCB1 = value + exploration_weight * sqrt(2 * ln(parent_visits) / child_visits)
        return max(self.children, key=lambda c: 
            c.value / c.visits + exploration_weight * math.sqrt(2 * math.log(self.visits) / c.visits))
    
    def expand(self):
        """未試行のアクションを一つ選び、新しい子ノードを追加"""
        action = self.untried_actions.pop()
        next_state = self.apply_action(self.state, action)
        child = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child
        
    def apply_action(self, state, action):
        """アクションを適用した新しい状態を返す"""
        # 深いコピーを作成して状態を変更
        new_state = deepcopy(state)
        # アクションに応じた状態変更
        # ここでは実際のゲームロジックに基づく状態更新を行う
        return new_state
        
    def update(self, result):
        """シミュレーション結果でノードを更新"""
        self.visits += 1
        self.value += result  # resultは勝ち=1, 負け=0のような値

class MCTSAgent(BaseAgent):
    def __init__(self, name="MCTS", simulation_count=100, exploration_weight=1.0):
        super().__init__(name)
        self.simulation_count = simulation_count
        self.exploration_weight = exploration_weight
        self.max_time = 1.0  # 最大計算時間（秒）
        
    def decide_action(self, player, simulation):
        """MCTSを使って最適なアクションを決定"""
        start_time = time.time()
        root = MCTSNode(self._extract_state(simulation, player))
        
        # 時間またはシミュレーション回数の制限まで実行
        simulation_count = 0
        while (time.time() - start_time < self.max_time and 
               simulation_count < self.simulation_count):
            # 選択: UCB1で最良の子ノードを選択
            node = root
            while node.untried_actions == [] and node.children != []:
                node = node.select_child(self.exploration_weight)
                
            # 拡張: 未試行のアクションがあれば子ノードを拡張
            if node.untried_actions != []:
                node = node.expand()
                
            # シミュレーション: ランダムプレイで終局まで進める
            result = self._simulate(node.state)
            
            # バックプロパゲーション: 結果を木の上に伝播
            while node is not None:
                node.update(result)
                node = node.parent
                
            simulation_count += 1
            
        # 最も訪問回数の多い子ノードのアクションを選択
        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        best_action = best_child.action if best_child else None
        
        self.record_action("mcts_decision", {
            "action": best_action,
            "simulations": simulation_count,
            "time": time.time() - start_time
        })
        
        return best_action
    
    def _extract_state(self, simulation, player):
        """シミュレーション状態のコピーを作成"""
        # ゲーム状態の重要な部分だけを抽出
        state = {
            "cities": deepcopy(simulation.cities),
            "players": deepcopy(simulation.players),
            "current_player": player,
            # その他必要な状態情報
        }
        return state
    
    def _simulate(self, state):
        """ランダムプレイによるシミュレーション"""
        # 単純な実装: ランダム行動でゲーム終了まで進め、勝敗を返す
        # 本来はゲームロジックに沿った実装が必要
        # ここでは簡易的に、50%の確率で勝利するとしておく
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
        print(f"MCTSエージェントの状態を{filepath}に保存しました")
        
    def load_state(self, filepath="mcts_agent_state.pkl"):
        """保存されたエージェントの状態を読み込み"""
        try:
            with open(filepath, "rb") as f:
                save_data = pickle.load(f)
                
            self.action_stats = save_data.get("action_stats", {})
            self.visit_counts = save_data.get("visit_counts", {})
            self.value_sum = save_data.get("value_sum", {})
            print(f"{filepath}からMCTSエージェントの状態を読み込みました")
            return True
        except:
            print(f"{filepath}の読み込みに失敗しました")
            return False
            
    def update_action_stats(self, state_hash, action, result):
        """行動統計情報を更新して学習を蓄積"""
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
        """アクションを辞書のキーに変換"""
        if not action:
            return "None"
        
        if action.get("type") == "move":
            city_name = action.get("target_city").name if action.get("target_city") else "unknown"
            return f"move-{city_name}"
        elif action.get("type") == "treat":
            city_name = action.get("city").name if action.get("city") else "current"
            return f"treat-{city_name}"
        # 他のアクションタイプも同様に...
        
        return str(action)

# グローバルエージェントインスタンス
_global_mcts_agent = None

# MCTS戦略関数
def mcts_agent_strategy(player):
    global _global_mcts_agent
    
    # 実験ディレクトリを取得
    import os
    log_dir = player.simulation.log_dir if hasattr(player.simulation, 'log_dir') else "./logs"
    state_file = os.path.join(log_dir, "mcts_agent_state.pkl")
    
    # デバッグ出力
    print(f"DEBUG: MCTS戦略関数が呼ばれました - エージェントは{_global_mcts_agent}")
    
    if _global_mcts_agent is None:
        _global_mcts_agent = MCTSAgent()
        print(f"新しいMCTSエージェントを作成（保存先: {state_file}）")
        # 状態を読み込む
        _global_mcts_agent.load_state(filepath=state_file)
    
    action = _global_mcts_agent.decide_action(player, player.simulation)
    
    # 定期的に保存（1%の確率で保存）
    if random.random() < 0.01:
        _global_mcts_agent.save_state(filepath=state_file)
    
    # 以下アクション変換（既存コード）...
    
    if not action:
        return None  # アクションなし
    
    # アクション実行コード（これを追加）
    if action.get("type") == "move":
        target_city = action.get("target_city")
        if target_city:
            return {"type": "move", "target": target_city}
    
    elif action.get("type") == "treat":
        target_city = action.get("city") or player.city
        if target_city.infection_level > 0:
            return {"type": "treat", "target": target_city}
    
    # その他のアクションタイプも同様に...
    
    # アクションが無効な場合
    return None