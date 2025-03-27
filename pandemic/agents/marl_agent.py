# agents/rl_agent.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
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
    def __init__(self, name="MARL", state_size=100, action_size=10, 
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
        
        self.epsilon = 1.0  # default ε-greedy
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # future reward discount
        
        self.train_count = 0
        
    def encode_state(self, player, simulation):
        state = np.zeros(self.state_size)
        
        city_idx = simulation.cities.index(player.city)
        state[city_idx] = 1
        
        for i, city in enumerate(simulation.cities):
            if i + len(simulation.cities) < self.state_size:
                state[i + len(simulation.cities)] = city.infection_level / 5.0  # normalize to 0-1 range
        
        for i, city in enumerate(simulation.cities):
            offset = 2 * len(simulation.cities)
            if i + offset < self.state_size:
                state[i + offset] = 1 if city.has_research_station else 0
                
        # TODO: add other features like player position, etc.
        
        return state
    
    def decode_action(self, action_idx, player, simulation):
        possible_actions = self._get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
            
        if action_idx < len(possible_actions):
            return possible_actions[action_idx]
        else:
            return random.choice(possible_actions)
        
    def remember(self, state, action, reward, next_state, done):
        """saves experience to memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """act based on ε-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        
        return np.argmax(action_values.numpy())
    
    def train(self, batch_size=32):
        """batch training from memory"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size) # rudom sampling
        
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
        state = self.encode_state(player, simulation)
        
        action_idx = self.act(state)
    
        action = self.decode_action(action_idx, player, simulation)
        
        next_simulation = self._simulate_action(simulation, player, action)
        
        next_state = self.encode_state(player, next_simulation)
        
        current_infection = sum(c.infection_level for c in simulation.cities)
        next_infection = sum(c.infection_level for c in next_simulation.cities)
        reward = (current_infection - next_infection) * 0.1
        
        done = all(c.infection_level == 0 for c in next_simulation.cities)
        if done:
            reward += 1.0
        
        self.remember(state, action_idx, reward, next_state, done)
        
        self.train()
        
        self.record_action("marl_decision", {
            "action": action,
            "epsilon": self.epsilon,
            "reward": reward
        })
        
        return action
    
    def _simulate_action(self, simulation, player, action):

        return simulation 
    
    def _get_possible_actions(self, player, simulation):
        actions = []
        
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

    def save_state(self, filepath="marl_agent_state.pt"):
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_count": self.train_count
        }
        
        torch.save(save_data, filepath)
        print(f"MARL agents state saved in {filepath}.")
        
    def load_state(self, filepath="marl_agent_state.pt"):
        try:
            if os.path.exists(filepath):
                save_data = torch.load(filepath)
                
                self.model.load_state_dict(save_data["model_state_dict"])
                self.target_model.load_state_dict(save_data["target_model_state_dict"])
                self.optimizer.load_state_dict(save_data["optimizer_state_dict"])
                self.epsilon = save_data.get("epsilon", self.epsilon)
                self.train_count = save_data.get("train_count", 0)
                
                print(f"loaded MARL state from {filepath}.")
                return True
                
            log_dirs = sorted([d for d in os.listdir("./logs") if d.startswith("experiment_")])
            if log_dirs:
                latest_log = log_dirs[-1]
                latest_path = f"./logs/{latest_log}/marl_agent_state.pt"
                if os.path.exists(latest_path):
                    print(f" loading MARL state file from {latest_path} as alternative. ")
                    save_data = torch.load(latest_path)
                    
                    self.model.load_state_dict(save_data["model_state_dict"])
                    self.target_model.load_state_dict(save_data["target_model_state_dict"])
                    self.optimizer.load_state_dict(save_data["optimizer_state_dict"])
                    self.epsilon = save_data.get("epsilon", self.epsilon)
                    self.train_count = save_data.get("train_count", 0)
                    
                    return True
                
            print("no existing state file found. creating new agent.")
            return False
        except Exception as e:
            print(f"error: {e}")
            return False

_global_marl_agent = None

def marl_agent_strategy(player):
    global _global_marl_agent
    
    import os
    log_dir = player.simulation.log_dir if hasattr(player.simulation, 'log_dir') else "./logs"
    state_file = os.path.join(log_dir, "marl_agent_state.pt")
    
    if _global_marl_agent is None:
        _global_marl_agent = MARLAgent()
        print(f"creating new MARL agent（saved in: {state_file}）")
        
        _global_marl_agent.load_state(filepath=state_file)
    
    action = _global_marl_agent.decide_action(player, player.simulation)
    
    if random.random() < 0.01:
        _global_marl_agent.save_state(filepath=state_file)
    
    if not action:
        return None
    
    if action.get("type") == "move":
        target_city = action.get("target_city")
        if target_city:
            return {"type": "move", "target": target_city}
    
    elif action.get("type") == "treat":
        city = action.get("city") or player.city
        if city.infection_level > 0:
            return {"type": "treat", "target": city}

    elif action.get("type") == "build":
        return {"type": "build", "target": player.city}
    
    return None