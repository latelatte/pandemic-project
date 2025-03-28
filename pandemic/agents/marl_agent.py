import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import os
from pandemic.agents.baseline_agents import BaseAgent

# define a tupe for experience 
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """add experience to new memory"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return [], [], []
            
        memory_size = len(self.memory)
        priorities = self.priorities[:memory_size]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(memory_size, batch_size, p=probs)
        
        weights = (memory_size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.memory[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, errors):
        """using TD error to update priorities"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

class DuelingDQNetwork(nn.Module):
    """Dueling DQN network"""
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DuelingDQNetwork, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class MARLAgent(BaseAgent):
    def __init__(self, name="MARL", state_size=100, action_size=20, 
                 memory_size=10000, learning_rate=0.0005):
        super().__init__(name)
        self.state_size = state_size
        self.action_size = action_size
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Dueling network based Double DQN
        self.model = DuelingDQNetwork(state_size, action_size)
        self.target_model = DuelingDQNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # discount factor
        
        self.update_frequency = 4
        self.target_update_frequency = 1000
        self.train_count = 0
    
    def encode_state(self, player, simulation):
        """encode the game state into a vector"""
        state = np.zeros(self.state_size)

        if player.city:
            city_idx = simulation.cities.index(player.city) if player.city in simulation.cities else 0
            state[city_idx] = 1
        
        # infectin level of each city
        offset = len(simulation.cities)
        for i, city in enumerate(simulation.cities):
            if i + offset < self.state_size:
                state[i + offset] = city.infection_level / 3.0
        
        # research station presence
        offset = 2 * len(simulation.cities)
        for i, city in enumerate(simulation.cities):
            if i + offset < self.state_size:
                state[i + offset] = 1 if city.has_research_station else 0
        
        # player's hand cards
        card_counts = {}
        for card in player.hand:
            if hasattr(card, 'city_name') and card.city_name:
                card_counts[card.city_name] = card_counts.get(card.city_name, 0) + 1
        
        for i, city in enumerate(simulation.cities):
            offset = 3 * len(simulation.cities)
            if i + offset < self.state_size:
                state[i + offset] = card_counts.get(city.name, 0) / 5.0
        
        # infection level and outbreak count
        outbreak_index = 4 * len(simulation.cities)
        if outbreak_index < self.state_size:
            state[outbreak_index] = simulation.outbreak_count / simulation.outbreak_limit
        
        infection_rate_index = outbreak_index + 1
        if infection_rate_index < self.state_size:
            state[infection_rate_index] = simulation.infection_rate / 4.0  # 正規化
        
        # discovered cures
        cure_index = infection_rate_index + 1
        if cure_index < self.state_size:
            state[cure_index] = len(simulation.discovered_cures) / 4.0  # 正規化
        
        # other players' cities
        other_players_offset = cure_index + 1
        player_index = simulation.players.index(player) if player in simulation.players else 0
        
        for i, p in enumerate(simulation.players):
            if p != player and other_players_offset + i < self.state_size:
                if p.city and p.city in simulation.cities:
                    other_city_idx = simulation.cities.index(p.city)
                    state[other_players_offset + i] = other_city_idx / len(simulation.cities)
        
        return state
    
    def calculate_reward(self, player, simulation, action, next_simulation):
        reward = 0.0
        
        # base reward - change in infection level
        
        # 1. change in infection level
        current_infection = sum(c.infection_level for c in simulation.cities)
        next_infection = sum(c.infection_level for c in next_simulation.cities)
        infection_change = current_infection - next_infection
        reward += infection_change * 0.5
        
        # 2. action-specific rewards
        action_type = action.get("type", "")
        
        if action_type == "build":
            # for research station
            if player.city and not player.city.has_research_station:
                reward += 2.0
        
        elif action_type == "treat":
            # for treating infection
            target = action.get("target") or player.city
            if target and target.infection_level > 0:
                reward += 0.5 * target.infection_level
                
                # prioritize high infection levels
                if target.infection_level >= 3:
                    reward += 1.0
        
        elif action_type == "move":
            target = action.get("target")
            if target:
                # moving to high infection level city
                if target.infection_level > 1:
                    reward += 0.2 * target.infection_level
                
                # moving to a city with research station
                if target.has_research_station:
                    reward += 0.3
        
        # 3. discovering cures
        if hasattr(simulation, 'discovered_cures') and hasattr(next_simulation, 'discovered_cures'):
            if len(next_simulation.discovered_cures) > len(simulation.discovered_cures):
                # successfully discovered a cure
                reward += 5.0
        
        # 4 . game over state
        if next_simulation.game_over:
            if next_simulation.is_win_condition():
                # wins
                reward += 10.0
            else:
                # loses
                reward -= 5.0
        
        # 5. bonus for cooperation
        if action_type == "move":
            target = action.get("target")
            if target:
                for other_player in simulation.players:
                    if other_player != player and other_player.city == target:
                        # move to the same city as another player
                        reward += 0.5
        
        return reward
    
    def decide_action(self, player, simulation):
        """decide action based on encoded state"""
        state = self.encode_state(player, simulation)
        
        # epsilon-greedy
        if random.random() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            action_idx = np.argmax(action_values.numpy())
        
        possible_actions = self._get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
            
        action = possible_actions[action_idx % len(possible_actions)]

        next_simulation = self.simulate_action(simulation, player, action)
        next_state = self.encode_state(player, next_simulation)
        
        reward = self.calculate_reward(player, simulation, action, next_simulation)

        done = next_simulation.game_over
        
        self.memory.add(state, action_idx, reward, next_state, done)
        
        if self.train_count % self.update_frequency == 0:
            self.train()
        
        self.train_count += 1
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.record_action("marl_decision", {
            "action": action,
            "epsilon": self.epsilon,
            "reward": reward
        })
        
        return action
    
    def _get_possible_actions(self, player, simulation):
        actions = []
        

        for neighbor in player.city.neighbours:
            actions.append({
                "type": "move",
                "target": neighbor,
                "method": "drive"
            })
            
        if player.city.infection_level > 0:
            actions.append({
                "type": "treat",
                "target": player.city
            })
            
        if not player.city.has_research_station:
            for card in player.hand:
                if hasattr(card, 'city_name') and card.city_name == player.city.name:
                    actions.append({
                        "type": "build",
                        "target": player.city
                    })
                    break
        
        return actions
    
    def simulate_action(self, simulation, player, action):
        from copy import deepcopy
        
        sim_copy = deepcopy(simulation)
        
        player_copy = None
        for p in sim_copy.players:
            if hasattr(p, 'id') and hasattr(player, 'id') and p.id == player.id:
                player_copy = p
                break
        
        if not player_copy:
            return sim_copy
        
        action_type = action.get("type")
        
        if action_type == "move":
            target = action.get("target")
            if target:
                for city in sim_copy.cities:
                    if city.name == target.name:
                        player_copy.city = city
                        break
        
        elif action_type == "treat":
            target = action.get("target")
            if target and target.infection_level > 0:
                for city in sim_copy.cities:
                    if city.name == target.name and city.infection_level > 0:
                        city.infection_level -= 1
                        break
        
        elif action_type == "build":
            if player_copy.city:
                player_copy.city.has_research_station = True
        
        return sim_copy
    
    def train(self, batch_size=64):
        batch, indices, weights = self.memory.sample(batch_size)
        if not batch:
            return
        
        states = torch.FloatTensor([exp.state for exp in batch])
        actions = torch.LongTensor([[exp.action] for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state for exp in batch])
        dones = torch.FloatTensor([exp.done for exp in batch])
        weights = torch.FloatTensor(weights)
        
        # Double DQN
        next_q_values = self.model(next_states)
        next_actions = next_q_values.max(1)[1].unsqueeze(1)
        
        with torch.no_grad():
            next_q_values_target = self.target_model(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones.unsqueeze(1)))
        
        current_q = self.model(states).gather(1, actions)
        
        td_errors = torch.abs(target_q - current_q).detach().squeeze().numpy()
        
        self.memory.update_priorities(indices, td_errors)
        
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.train_count % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def save_state(self, filepath="marl_agent_state.pt"):
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_count": self.train_count
        }
        
        torch.save(save_data, filepath)
        print(f"MARL agent state saved in {filepath} (epsilon: {self.epsilon:.4f}, train_count: {self.train_count})") # for debugging
    
    def load_state(self, filepath="marl_agent_state.pt"):
        try:
            if os.path.exists(filepath):
                save_data = torch.load(filepath)
                
                self.model.load_state_dict(save_data["model_state_dict"])
                self.target_model.load_state_dict(save_data["target_model_state_dict"])
                self.optimizer.load_state_dict(save_data["optimizer_state_dict"])
                self.epsilon = save_data.get("epsilon", self.epsilon)
                self.train_count = save_data.get("train_count", 0)
                
                print(f"Loaded MARL state from {filepath}")
                return True
            
            return False
        except Exception as e:
            print(f"Error loading state: {e}")
            return False

_global_marl_agent = None

def marl_agent_strategy(player):
    global _global_marl_agent
    
    import os
    agent_state_dir = "./agents_state"
    os.makedirs(agent_state_dir, exist_ok=True)
    state_file = os.path.join(agent_state_dir, "marl_agent_state.pt")
    
    if _global_marl_agent is None:
        _global_marl_agent = MARLAgent()
        print(f"Created new MARL agent (state file: {state_file})")
        _global_marl_agent.load_state(state_file)
    
    action = _global_marl_agent.decide_action(player, player.simulation)

    if random.random() < 0.01:
        _global_marl_agent.save_state(state_file)
    
    return action