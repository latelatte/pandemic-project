import math
import random
import time
from copy import deepcopy
from pandemic.agents.baseline_agents import BaseAgent

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
        # state（ゲーム状態）から可能なアクションを取得
        # ここではシミュレーションの状態から可能なアクションを取得する関数を呼び出すイメージ
        return []  # 実際の実装ではゲームロジックに基づいて取得
        
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

# MCTS戦略関数
def mcts_agent_strategy(player):
    agent = MCTSAgent()
    while player.actions_remaining > 0:
        action = agent.decide_action(player, player.simulation)
        if not action:
            break
            
        # アクション実行
        # ...アクションのタイプに応じた処理
        
        player.actions_remaining -= 1