# agents/rl_agent.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
from pandemic.agents.baseline_agents import BaseAgent

# Neural network model for DQN
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
    def __init__(self, name="MARL", state_size=100, action_size=20, 
                 memory_size=2000, learning_rate=0.001):
        super().__init__(name)
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=memory_size)
        
        # Q-Network
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor
        
        self.train_count = 0
        
    def encode_state(self, player, simulation):
        """
        Encode the current game state into a feature vector.
        
        Args:
            player: Current player
            simulation: Current simulation state
            
        Returns:
            numpy array representing the state
        """
        state = np.zeros(self.state_size)
        
        # 名前ベースで都市を検索
        city_idx = 0  # デフォルト：最初の都市
        if player.city:
            for i, city in enumerate(simulation.cities):
                if city.name == player.city.name:
                    city_idx = i
                    break
        
        state[city_idx] = 1
        
        # 感染レベル
        for i, city in enumerate(simulation.cities):
            offset = len(simulation.cities)
            if i + offset < self.state_size:
                state[i + offset] = city.infection_level / 3.0  # Normalize to 0-1 range
        
        # 研究所の位置
        for i, city in enumerate(simulation.cities):
            offset = 2 * len(simulation.cities)
            if i + offset < self.state_size:
                state[i + offset] = 1 if city.has_research_station else 0
        
        # プレイヤーの手札サイズ
        offset = 3 * len(simulation.cities)
        if offset < self.state_size:
            state[offset] = len(player.hand) / 7.0  # Normalize assuming max hand size is 7
        
        # アウトブレイク数
        offset = 3 * len(simulation.cities) + 1
        if offset < self.state_size:
            state[offset] = simulation.outbreak_count / 8.0  # Normalize assuming max outbreaks is 8
        
        # 感染率
        offset = 3 * len(simulation.cities) + 2
        if offset < self.state_size:
            infection_rate = getattr(simulation, 'infection_rate', 2)  # Default if not found
            state[offset] = infection_rate / 4.0
        
        # 病気の治癒状態
        if hasattr(simulation, 'diseases'):
            for i, disease in enumerate(simulation.diseases):
                offset = 3 * len(simulation.cities) + 3 + i
                if offset < self.state_size:
                    if isinstance(disease, dict):
                        disease_cured = disease.get("cured", False)
                    else:
                        disease_cured = getattr(disease, "cured", False)
                    state[offset] = 1 if disease_cured else 0
        
        # 他プレイヤーの位置 - 名前ベースで検索
        for i, p in enumerate(simulation.players):
            # プレイヤーを名前で識別（id属性がない場合の対策）
            if hasattr(p, 'name') and hasattr(player, 'name') and p.name != player.name:
                offset = 3 * len(simulation.cities) + 7 + i
                if offset < self.state_size:
                    # 都市のインデックスを名前ベースで取得
                    city_idx = 0
                    if p.city:
                        for j, city in enumerate(simulation.cities):
                            if city.name == p.city.name:
                                city_idx = j
                                break
                    state[offset] = city_idx / len(simulation.cities)
        
        return state
    
    def decode_action(self, action_idx, player, simulation):
        """
        Convert action index to actual game action
        
        Args:
            action_idx: Index of the selected action
            player: Current player
            simulation: Current simulation state
            
        Returns:
            Action dictionary or None if invalid
        """
        possible_actions = self.get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
            
        if action_idx < len(possible_actions):
            return possible_actions[action_idx]
        else:
            return random.choice(possible_actions)
        
    def remember(self, state, action, reward, next_state, done):
        """
        Save experience tuple to replay memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action index
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        
        return np.argmax(action_values.numpy())
    
    def train(self, batch_size=32):
        """
        Train the Q-network with a batch of experiences
        
        Args:
            batch_size: Number of samples to use for training
        """
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(
                        self.target_model(next_state_tensor)).item()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            current_q = self.model(state_tensor)
            
            target_q = current_q.clone().detach()
            target_q[0][action] = target
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q, target_q)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.train_count += 1
        if self.train_count % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def decide_action(self, player, simulation):
        """
        Main decision method that selects and returns an action
        
        Args:
            player: Current player
            simulation: Current simulation state
            
        Returns:
            Selected action dictionary
        """
        state = self.encode_state(player, simulation)
        
        action_idx = self.act(state)
    
        action = self.decode_action(action_idx, player, simulation)
        
        # actionがNoneの場合にはデフォルトアクションを使用
        if action is None:
            print("Warning: No valid action found in MARL agent. Using pass action.")
            action = {"type": "pass"}  # 何もしないアクション
        
        try:
            # simulate_actionを安全に呼び出す
            next_simulation = self.simulate_action(simulation, player, action)
            
            # シミュレーション結果がNoneの場合のフォールバック
            if next_simulation is None:
                print("Warning: Simulation returned None, using original simulation")
                next_simulation = simulation  # 元のシミュレーションを使用
        except Exception as e:
            print(f"Simulation error: {e}")
            next_simulation = simulation  # エラー時は元のシミュレーションを使用
        
        next_state = self.encode_state(player, next_simulation)
        
        current_infection = sum(c.infection_level for c in simulation.cities)
        next_infection = sum(c.infection_level for c in next_simulation.cities)
        reward = (current_infection - next_infection) * 0.1

        research_stations_before = sum(1 for c in simulation.cities if c.has_research_station)
        research_stations_after = sum(1 for c in next_simulation.cities if c.has_research_station)
        reward += (research_stations_after - research_stations_before) * 0.5
        
        if hasattr(simulation, 'diseases'):
            diseases_cured_before = 0
            for disease in simulation.diseases:
                if isinstance(disease, dict):
                    if disease.get("cured", False):
                        diseases_cured_before += 1
                else:
                    if getattr(disease, "cured", False):
                        diseases_cured_before += 1
        else:
            diseases_cured_before = 0

        if hasattr(next_simulation, 'diseases'):
            diseases_cured_after = 0
            for disease in next_simulation.diseases:
                if isinstance(disease, dict):
                    if disease.get("cured", False):
                        diseases_cured_after += 1
                else:
                    if getattr(disease, "cured", False):
                        diseases_cured_after += 1
        else:
            diseases_cured_after = 0

        reward += (diseases_cured_after - diseases_cured_before) * 2.0
        
        # Check if victory
        if hasattr(simulation, 'diseases'):
            all_diseases_cured = True
            for disease in simulation.diseases:
                if isinstance(disease, dict):
                    if not disease.get("cured", False):
                        all_diseases_cured = False
                        break
                else:
                    if not getattr(disease, "cured", False):
                        all_diseases_cured = False
                        break
            done = all_diseases_cured
        else:
            done = False
            
        if done:
            reward += 10.0
        
        self.remember(state, action_idx, reward, next_state, done)
        
        self.train()
        
        self.record_action("marl_decision", {
            "action": action,
            "epsilon": self.epsilon,
            "reward": reward
        })
        
        return action
    
    def simulate_action(self, simulation, player, action):
        """シミュレーション関数の完全実装"""
        from copy import deepcopy
        
        sim_copy = deepcopy(simulation)
        
        # プレイヤーを名前で検索
        player_copy = None
        for p in sim_copy.players:
            if hasattr(p, 'name') and hasattr(player, 'name') and p.name == player.name:
                player_copy = p
                break
        
        if not player_copy:
            print("Warning: Player copy not found in simulation.")
            return sim_copy
        
        action_type = action.get("type")
        
        if action_type == "move":
            target_city = action.get("target_city")
            method = action.get("method", "standard")
            card = action.get("card")
            
            if target_city:
                # 対応する都市をシミュレーションコピー内で見つける
                for city in sim_copy.cities:
                    if city.name == target_city.name:
                        player_copy.city = city
                        break
                
                # 移動方法に応じたカード処理
                if method in ["direct_flight", "charter_flight"] and card:
                    for c in list(player_copy.hand):
                        if hasattr(c, 'city') and c.city and hasattr(card, 'city') and card.city and c.city.name == card.city.name:
                            player_copy.hand.remove(c)
                            break
        
        elif action_type == "treat":
            city = action.get("city") or player_copy.city
            color = action.get("color", "Blue")  # デフォルト色
            
            # 感染レベルを下げる
            if city.infection_level > 0:
                city.infection_level -= 1
                
                # 色指定がある場合は疾病キューブも減らす
                if hasattr(city, 'disease_cubes') and isinstance(city.disease_cubes, dict) and color in city.disease_cubes:
                    city.disease_cubes[color] = max(0, city.disease_cubes[color] - 1)
        
        elif action_type == "build":
            # 研究所を建設
            if player_copy.city:
                player_copy.city.has_research_station = True
                
                # カード処理
                city_card = None
                for c in list(player_copy.hand):
                    if hasattr(c, 'city') and c.city and c.city.name == player_copy.city.name:
                        city_card = c
                        break
                
                if city_card:
                    player_copy.hand.remove(city_card)
        
        elif action_type == "share_knowledge":
            # 知識共有の実装
            direction = action.get("direction")
            target_player = action.get("target_player")
            card = action.get("card")
            
            # ターゲットプレイヤーを検索
            target_player_copy = None
            for p in sim_copy.players:
                if hasattr(p, 'name') and hasattr(target_player, 'name') and p.name == target_player.name:
                    target_player_copy = p
                    break
            
            if target_player_copy and card:
                source = player_copy if direction == "give" else target_player_copy
                recipient = target_player_copy if direction == "give" else player_copy
                
                # カードを探して移動
                for c in list(source.hand):
                    if hasattr(c, 'city') and c.city and hasattr(card, 'city') and card.city and c.city.name == card.city.name:
                        source.hand.remove(c)
                        recipient.hand.append(c)
                        break
        
        elif action_type == "discover_cure":
            # 治療法発見の実装 - 重要アクション
            color = action.get("color")
            cards = action.get("cards", [])
            
            # 研究所が必要
            if player_copy.city and player_copy.city.has_research_station:
                # 十分なカードがあるか確認
                required_cards = 5
                if hasattr(player_copy, 'role') and player_copy.role == "Scientist":
                    required_cards = 4
                
                if len(cards) >= required_cards:
                    # 疾病を治療済みに設定
                    if hasattr(sim_copy, 'diseases'):
                        for disease in sim_copy.diseases:
                            if isinstance(disease, dict):
                                if disease.get("color") == color:
                                    disease["cured"] = True
                            elif hasattr(disease, 'color') and disease.color == color:
                                disease.cured = True
                    
                    # カードを使用
                    cards_to_discard = []
                    for i in range(min(required_cards, len(cards))):
                        card = cards[i]
                        for c in list(player_copy.hand):
                            if hasattr(c, 'city') and c.city and hasattr(card, 'city') and card.city and c.city.name == card.city.name:
                                cards_to_discard.append(c)
                                break
                    
                    # 手札から削除
                    for c in cards_to_discard:
                        if c in player_copy.hand:
                            player_copy.hand.remove(c)
        
        elif action_type == "pass":
            # 何もしない
            pass
        
        return sim_copy
    
    def save_state(self, filepath="marl_agent_state.pt"):
        """
        Save agent state to file
        
        Args:
            filepath: Path to save file
        """
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_count": self.train_count
        }
        
        torch.save(save_data, filepath)
        
    def load_state(self, filepath="marl_agent_state.pt"):
        """
        Load agent state from file
        
        Args:
            filepath: Path to load file
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if os.path.exists(filepath):
                save_data = torch.load(filepath)
                
                # 念のためキーの存在確認
                if "model_state_dict" in save_data and "target_model_state_dict" in save_data:
                    self.model.load_state_dict(save_data["model_state_dict"])
                    self.target_model.load_state_dict(save_data["target_model_state_dict"])
                    
                    if "optimizer_state_dict" in save_data:
                        self.optimizer.load_state_dict(save_data["optimizer_state_dict"])
                        
                    self.epsilon = save_data.get("epsilon", self.epsilon)
                    self.train_count = save_data.get("train_count", 0)
                    
                    print(f"Loaded MARL state from {filepath} (epsilon: {self.epsilon:.4f}, train_count: {self.train_count})")
                    return True
                else:
                    print(f"Warning: Invalid format in state file {filepath}")
                    return False
            
            return False
        except Exception as e:
            print(f"Error loading state: {e}")
            return False


_global_marl_agent = None

def marl_agent_strategy(player):
    """
    Strategy function that interfaces with the game engine
    
    Args:
        player: Current player
        
    Returns:
        Action in the game engine's format
    """
    global _global_marl_agent
    
    import os
    
    agent_state_dir = "./agents_state"
    os.makedirs(agent_state_dir, exist_ok=True)
    state_file = os.path.join(agent_state_dir, "marl_agent_state.pt")
    
    if _global_marl_agent is None:
        if os.path.exists(state_file):
            try:
                _global_marl_agent = MARLAgent()
                loaded = _global_marl_agent.load_state(filepath=state_file)
                if loaded:
                    print(f"Successfully loaded MARL agent state from {state_file}")
                else:
                    print(f"Failed to load MARL agent state. Starting fresh.")
                    _global_marl_agent = MARLAgent()
            except Exception as e:
                print(f"Error loading agent state: {e}")
                _global_marl_agent = MARLAgent()
        else:
            print(f"No state file found. Creating new MARL agent.")
            _global_marl_agent = MARLAgent()
    
    action = _global_marl_agent.decide_action(player, player.simulation)

    
    # アクション変換（以下同じ）...
    if not action:
        return None
    
    # Convert internal action representation to game engine format
    if action.get("type") == "move":
        target_city = action.get("target_city")
        if target_city:
            return {"type": "move", "target": target_city}
    
    elif action.get("type") == "treat":
        city = action.get("city") or player.city
        color = action.get("color")
        return {"type": "treat", "target": city, "color": color} if color else {"type": "treat", "target": city}

    elif action.get("type") == "build":
        return {"type": "build", "target": player.city}
    
    elif action.get("type") == "share_knowledge":
        target_player = action.get("target_player")
        card = action.get("card")
        direction = action.get("direction")
        
        if target_player and card:
            return {
                "type": "share_knowledge",
                "target_player": target_player,
                "card": card,
                "direction": direction
            }
    
    elif action.get("type") == "discover_cure":
        color = action.get("color")
        cards = action.get("cards")
        
        if color and cards:
            return {
                "type": "discover_cure",
                "color": color,
                "cards": cards
            }
    
    return None