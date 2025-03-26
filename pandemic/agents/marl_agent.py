# agents/rl_agent.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pandemic.agents.baseline_agents import BaseAgent

# ニューラルネットワークモデル
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MARLAgent(BaseAgent):
    """マルチエージェント強化学習を使用するエージェント"""
    def __init__(self, name="MARL", state_size=100, action_size=10, 
                 memory_size=2000, learning_rate=0.001):
        super().__init__(name)
        self.state_size = state_size
        self.action_size = action_size
        
        # 経験再生用のメモリバッファ
        self.memory = deque(maxlen=memory_size)
        
        # Q-Network
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # オプティマイザー
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 探索パラメータ
        self.epsilon = 1.0  # 初期値: 100%探索
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # 割引率
        
        # 学習カウンター
        self.train_count = 0
        
    def encode_state(self, player, simulation):
        """ゲーム状態をニューラルネット入力用にエンコード"""
        state = np.zeros(self.state_size)
        
        # プレイヤーの位置（都市ID）
        city_idx = simulation.cities.index(player.city)
        state[city_idx] = 1
        
        # 都市の感染レベル
        for i, city in enumerate(simulation.cities):
            if i + len(simulation.cities) < self.state_size:
                state[i + len(simulation.cities)] = city.infection_level / 5.0  # 正規化
        
        # 研究所の存在
        for i, city in enumerate(simulation.cities):
            offset = 2 * len(simulation.cities)
            if i + offset < self.state_size:
                state[i + offset] = 1 if city.has_research_station else 0
                
        # その他の状態情報も追加可能
        
        return state
    
    def decode_action(self, action_idx, player, simulation):
        """アクションインデックスを実際のアクションに変換"""
        possible_actions = self._get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
            
        if action_idx < len(possible_actions):
            return possible_actions[action_idx]
        else:
            return random.choice(possible_actions)
        
    def remember(self, state, action, reward, next_state, done):
        """経験をメモリに保存"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """状態に基づいて行動選択（ε-greedy方式）"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        # ニューラルネットで行動価値を予測
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        
        return np.argmax(action_values.numpy())
    
    def train(self, batch_size=32):
        """経験からバッチ学習"""
        if len(self.memory) < batch_size:
            return
            
        # バッチをランダムサンプリング
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(
                        self.target_model(next_state_tensor)).item()
            
            # 現在のQ値を取得
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            current_q = self.model(state_tensor)
            
            # ターゲットQ値を設定
            target_q = current_q.clone().detach()
            target_q[0][action] = target
            
            # モデルを更新
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q, target_q)
            loss.backward()
            self.optimizer.step()
            
        # εを減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # 定期的にターゲットネットワークを更新
        self.train_count += 1
        if self.train_count % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def decide_action(self, player, simulation):
        """強化学習を使って最適なアクションを決定"""
        # 現在の状態をエンコード
        state = self.encode_state(player, simulation)
        
        # 行動選択 (ε-greedy)
        action_idx = self.act(state)
        
        # 行動のデコード
        action = self.decode_action(action_idx, player, simulation)
        
        # 行動を実行してみる（シミュレーションのコピーで）
        next_simulation = self._simulate_action(simulation, player, action)
        
        # 次の状態をエンコード
        next_state = self.encode_state(player, next_simulation)
        
        # 報酬を計算（実装例: 感染レベル減少でプラス、増加でマイナス）
        current_infection = sum(c.infection_level for c in simulation.cities)
        next_infection = sum(c.infection_level for c in next_simulation.cities)
        reward = (current_infection - next_infection) * 0.1
        
        # ゲーム終了条件（全都市で感染レベル0）
        done = all(c.infection_level == 0 for c in next_simulation.cities)
        if done:
            reward += 1.0  # 勝利ボーナス
        
        # 経験を保存
        self.remember(state, action_idx, reward, next_state, done)
        
        # 定期的に学習
        self.train()
        
        self.record_action("marl_decision", {
            "action": action,
            "epsilon": self.epsilon,
            "reward": reward
        })
        
        return action
    
    def _simulate_action(self, simulation, player, action):
        """アクションの効果をシミュレーション"""
        # 簡易的な実装: 実際のゲームロジックに基づいて状態遷移を行う
        # 深いコピーを使用するとリソースを大量に消費するため注意
        return simulation  # 単純化のため元の状態を返す
    
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

# MARL戦略関数
def marl_agent_strategy(player):
    agent = MARLAgent()
    action = agent.decide_action(player, player.simulation)
    
    if not action:
        return None  # アクションなし
    
    # アクション情報をPlayer.perform_turnが理解できる形式に変換
    if action.get("type") == "move":
        target_city = action.get("target_city")
        if target_city:
            return {"type": "move", "target": target_city}
    
    elif action.get("type") == "treat":
        city = action.get("city") or player.city
        if city.infection_level > 0:
            return {"type": "treat", "target": city}
    
    # その他のアクション...
    # 例えば研究所建設など
    elif action.get("type") == "build":
        return {"type": "build", "target": player.city}
    
    # アクションが無効な場合
    return None