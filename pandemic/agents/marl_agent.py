import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import deque, namedtuple, defaultdict, Counter
import time

# Define a tuple for experience 
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for efficient learning from important experiences"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to memory with maximum priority for new entries"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of experiences with importance sampling weights"""
        if len(self.memory) < batch_size:
            return [], [], []
            
        memory_size = len(self.memory)
        priorities = self.priorities[:memory_size]
        
        # Convert priorities to sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample according to priorities
        indices = np.random.choice(memory_size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (memory_size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        # Increase beta toward 1 for less bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.memory[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

class RelationalAttention(nn.Module):
    """Attention mechanism to capture relationships between cities and players"""
    def __init__(self, feature_dim, num_heads=4):
        super(RelationalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Project input to query, key, value
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose to get dimensions: [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attention, v)
        
        # Transpose and reshape: [batch_size, seq_length, feature_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.feature_dim)
        
        # Final linear projection
        out = self.fc_out(out)
        
        return out

class StrategicNetworkArchitecture(nn.Module):
    """Enhanced network architecture with attention mechanism and strategic decision components"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(StrategicNetworkArchitecture, self).__init__()
        
        # Feature extraction layers
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Attention mechanism for relational reasoning
        self.attention = RelationalAttention(hidden_size)
        
        # Strategic goal-specific heads
        self.infection_control_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        self.cure_discovery_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        self.cooperation_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(3 * (hidden_size // 2), hidden_size // 2),  # Combines all strategic heads
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_extraction(x)
        
        # Add batch dimension if needed
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
            
        # Reshape for attention: [batch_size, sequence_length, hidden_size]
        # In this case, sequence_length = 1 as we're processing a single state
        features = features.unsqueeze(1)
        
        # Apply attention
        attentive_features = self.attention(features).squeeze(1)
        
        # Process through strategic heads
        infection_features = self.infection_control_head(attentive_features)
        cure_features = self.cure_discovery_head(attentive_features)
        cooperation_features = self.cooperation_head(attentive_features)
        
        # Combine strategic heads
        combined_strategic_features = torch.cat([
            infection_features, 
            cure_features, 
            cooperation_features
        ], dim=1)
        
        # Compute state value
        value = self.value_stream(attentive_features)
        
        # Compute action advantages
        advantages = self.advantage_stream(combined_strategic_features)
        
        # Combine value and advantages (Dueling Q-Network architecture)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class ImprovedMARLAgent:
    """
    Enhanced Multi-Agent Reinforcement Learning agent for Pandemic game.
    Features strategic state representation, cooperative reward design, and relational reasoning.
    """
    def __init__(self, name="MARL", state_size=200, action_size=50, 
                 memory_size=50000, batch_size=64, learning_rate=0.0003,
                 gamma=0.99, tau=0.005):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Initialize networks
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_net = StrategicNetworkArchitecture(state_size, action_size).to(self.device)
        self.target_net = StrategicNetworkArchitecture(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target net to evaluation mode
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Training parameters
        self.update_frequency = 4
        self.target_update_frequency = 1000
        self.train_count = 0
        
        # Game state tracking
        self.previous_cities = {}
        self.movement_history = {}
        self.infection_hotspots = []
        self.team_strategy = {}
        self.cure_progress = defaultdict(float)
        
        # Action statistics
        self.action_stats = {}
        
        # Performance metrics
        self.rewards_history = []
        self.episode_rewards = 0
        self.total_train_steps = 0
    
    def encode_strategic_state(self, player, simulation):
        """
        Enhanced state encoding that captures strategic game elements
        including infection risks, cure progress, and cooperation opportunities.
        """
        state = np.zeros(self.state_size)
        
        # Section 1: Player position (one-hot encoding)
        city_offset = 0
        if player.city and player.city in simulation.cities:
            city_idx = simulation.cities.index(player.city)
            if city_idx + city_offset < self.state_size:
                state[city_idx + city_offset] = 1
        
        # Section 2: Infection levels of cities
        infection_offset = len(simulation.cities)
        for i, city in enumerate(simulation.cities):
            if i + infection_offset < self.state_size:
                state[i + infection_offset] = city.infection_level / 3.0  # Normalized infection level
        
        # Section 3: Research station locations
        station_offset = 2 * len(simulation.cities)
        for i, city in enumerate(simulation.cities):
            if i + station_offset < self.state_size:
                state[i + station_offset] = 1 if city.has_research_station else 0
        
        # Section 4: Player's cards (by city)
        card_offset = 3 * len(simulation.cities)
        city_cards = {}
        for card in player.hand:
            if hasattr(card, 'city_name') and card.city_name:
                city_cards[card.city_name] = city_cards.get(card.city_name, 0) + 1
        
        for i, city in enumerate(simulation.cities):
            if i + card_offset < self.state_size:
                state[i + card_offset] = city_cards.get(city.name, 0) / 5.0  # Normalized card count
        
        # Section 5: Cards by color (for cure discovery assessment)
        color_offset = 4 * len(simulation.cities)
        color_counts = {"Blue": 0, "Red": 0, "Yellow": 0, "Black": 0}
        
        for card in player.hand:
            if hasattr(card, 'color') and card.color in color_counts:
                color_counts[card.color] += 1
        
        for i, (color, count) in enumerate(color_counts.items()):
            idx = color_offset + i
            if idx < self.state_size:
                state[idx] = count / 5.0  # Normalized color count
        
        # Section 6: Game state indicators
        game_state_offset = color_offset + len(color_counts)
        
        # Outbreak level
        outbreak_idx = game_state_offset
        if outbreak_idx < self.state_size:
            state[outbreak_idx] = simulation.outbreak_count / simulation.outbreak_limit
        
        # Infection rate
        infection_rate_idx = outbreak_idx + 1
        if infection_rate_idx < self.state_size:
            state[infection_rate_idx] = getattr(simulation, 'infection_rate', 2) / 4.0
        
        # Discovered cures
        cure_idx = infection_rate_idx + 1
        if cure_idx < self.state_size:
            state[cure_idx] = len(getattr(simulation, 'discovered_cures', [])) / 4.0
        
        # Section 7: Other players' positions and cards
        other_players_offset = cure_idx + 1
        other_player_idx = 0
        
        for other_player in simulation.players:
            if other_player != player:
                # Other player's position
                if other_player.city and other_player.city in simulation.cities:
                    city_idx = simulation.cities.index(other_player.city)
                    pos_idx = other_players_offset + (other_player_idx * 2)
                    if pos_idx < self.state_size:
                        state[pos_idx] = city_idx / len(simulation.cities)
                
                # Other player's number of cards
                cards_idx = other_players_offset + (other_player_idx * 2) + 1
                if cards_idx < self.state_size:
                    state[cards_idx] = len(other_player.hand) / 7.0  # Normalized by max hand size
                
                other_player_idx += 1
        
        # Section 8: Infection risk assessment
        risk_offset = other_players_offset + (len(simulation.players) * 2)
        
        # Identify high-risk cities (infection level 2+)
        high_risk_cities = []
        for city in simulation.cities:
            if city.infection_level >= 2:
                high_risk_cities.append(city)
        
        # Store infection hotspots for decision making
        self.infection_hotspots = sorted(high_risk_cities, 
                                       key=lambda city: city.infection_level,
                                       reverse=True)
        
        # Encode top 3 highest infection cities
        for i, city in enumerate(high_risk_cities[:3]):  # Limit to top 3
            if city and city in simulation.cities:
                city_idx = simulation.cities.index(city)
                idx = risk_offset + i
                if idx < self.state_size:
                    state[idx] = city_idx / len(simulation.cities)
                
                # Also encode infection level
                level_idx = risk_offset + 3 + i
                if level_idx < self.state_size:
                    state[level_idx] = city.infection_level / 3.0
        
        # Section 9: Cure discovery progress
        cure_progress_offset = risk_offset + 6
        
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        colors = ["Blue", "Red", "Yellow", "Black"]
        
        for i, color in enumerate(colors):
            progress = 1.0 if color in discovered_cures else 0.0
            
            if not progress:
                # Count cards of this color across all players
                total_cards = sum(1 for p in simulation.players
                               for card in p.hand
                               if hasattr(card, 'color') and card.color == color)
                
                # Approximate progress (5 cards needed, scientist needs 4)
                scientist_present = any(getattr(p, 'role', None) and 
                                      getattr(p.role, 'name', '') == "Scientist"
                                      for p in simulation.players)
                
                cards_needed = 4 if scientist_present else 5
                progress = min(0.99, total_cards / (cards_needed * 1.5))  # Scale to account for card distribution
            
            idx = cure_progress_offset + i
            if idx < self.state_size:
                state[idx] = progress
                self.cure_progress[color] = progress
        
        # Section 10: Cooperation opportunities
        coop_offset = cure_progress_offset + len(colors)
        
        # Encode if other players are in same city (potential for card sharing)
        same_city_count = sum(1 for p in simulation.players if p != player and p.city == player.city)
        if coop_offset < self.state_size:
            state[coop_offset] = min(1.0, same_city_count / (len(simulation.players) - 1))
        
        # Encode if player has city card matching current location (can be shared)
        has_current_city_card = any(hasattr(card, 'city_name') and card.city_name == player.city.name
                                  for card in player.hand)
        if coop_offset + 1 < self.state_size:
            state[coop_offset + 1] = 1.0 if has_current_city_card else 0.0
        
        # Player's role (one-hot encoding)
        role_offset = coop_offset + 2
        role_types = ["Medic", "Scientist", "Operations_Expert", "Researcher", "Dispatcher"]
        
        player_role = getattr(player, 'role', None)
        player_role_name = getattr(player_role, 'name', '') if player_role else ''
        
        for i, role in enumerate(role_types):
            idx = role_offset + i
            if idx < self.state_size:
                state[idx] = 1.0 if player_role_name == role else 0.0
        
        return state
    
    def select_action(self, state, available_actions, evaluation=False):
        """
        Select an action using epsilon-greedy policy with available actions filtering
        """
        if not available_actions:
            return None
            
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon or evaluation:
            with torch.no_grad():
                # Get Q-values for all actions
                q_values = self.policy_net(state_tensor)
                
                # Create valid action mask (initialize with -inf)
                action_mask = torch.ones(self.action_size) * float('-inf')
                
                # Set valid actions to their actual Q-values
                for i, action in enumerate(available_actions):
                    action_idx = self._action_to_index(action)
                    if action_idx < self.action_size:
                        action_mask[action_idx] = q_values[0, action_idx]
                
                # Get action with highest Q-value among valid actions
                action_idx = torch.argmax(action_mask).item()
                
                # Map back to actual action
                selected_action = None
                for action in available_actions:
                    if self._action_to_index(action) == action_idx:
                        selected_action = action
                        break
                
                # Fallback to random if mapping failed
                if selected_action is None:
                    selected_action = random.choice(available_actions)
                    
                return selected_action
        else:
            # Random action selection during exploration
            return random.choice(available_actions)
    
    def _action_to_index(self, action):
        """
        Convert an action dictionary to a unique index for the network
        """
        action_type = action.get("type", "unknown")
        
        # Basic mapping of action types to index ranges
        type_ranges = {
            "move": (0, 25),       # First 25 indices for movement actions
            "treat": (25, 35),     # Next 10 indices for treatment actions
            "build": (35, 40),     # Next 5 indices for building research stations
            "discover_cure": (40, 44),  # 4 indices for discovering cures (one per color)
            "share_knowledge": (44, 48), # 4 indices for sharing knowledge
            "pass": (49, 50)       # Last index for pass action
        }
        
        if action_type not in type_ranges:
            return 49  # Default to pass action index
            
        base_idx, _ = type_ranges[action_type]
        
        # Further distinguish within action types
        if action_type == "move":
            target_city = action.get("target_city")
            if target_city:
                # Use mod to handle more cities than available indices
                return base_idx + (hash(target_city.name) % 25)
        
        elif action_type == "treat":
            city = action.get("city")
            if city:
                return base_idx + (hash(city.name) % 10)
        
        elif action_type == "discover_cure":
            color = action.get("color")
            color_idx = {"Blue": 0, "Red": 1, "Yellow": 2, "Black": 3}.get(color, 0)
            return base_idx + color_idx
        
        elif action_type == "share_knowledge":
            direction = action.get("direction", "give")
            direction_idx = 0 if direction == "give" else 1
            return base_idx + direction_idx
        
        return base_idx  # Default to base index for the action type
    
    def _update_team_strategy(self, simulation):
        """Update team strategy based on game state"""
        # Count discovered cures
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        cure_count = len(discovered_cures)
        
        # Calculate outbreak risk
        outbreak_risk = simulation.outbreak_count / simulation.outbreak_limit
        
        # Assess infection pressure
        high_infection_cities = sum(1 for city in simulation.cities 
                                  if getattr(city, 'infection_level', 0) >= 2)
        infection_pressure = high_infection_cities / max(1, len(simulation.cities) * 0.1)
        
        # Determine overall strategy
        if cure_count >= 3:
            # Late game: focus on final cure
            self.team_strategy = {
                "priority": "final_cure",
                "secondary": "control_critical" if outbreak_risk > 0.5 else "cure_progress"
            }
        elif outbreak_risk > 0.7 or infection_pressure > 0.7:
            # Crisis management: prevent loss
            self.team_strategy = {
                "priority": "control_outbreaks",
                "secondary": "cure_progress"
            }
        elif cure_count == 0:
            # Early game: establish infrastructure and control spread
            self.team_strategy = {
                "priority": "research_infrastructure" if outbreak_risk < 0.3 else "control_spread",
                "secondary": "cure_preparation"
            }
        else:
            # Mid game: balance priorities
            self.team_strategy = {
                "priority": "cure_progress" if outbreak_risk < 0.5 else "control_critical",
                "secondary": "optimize_positioning"
            }
    
    def calculate_strategic_reward(self, player, simulation, action, next_simulation):
        """
        Enhanced reward function that considers strategic goals, long-term progress,
        and cooperative behavior.
        """
        # Base reward starts neutral
        reward = 0.0
        action_type = action.get("type", "")
        
        # 1. Win/loss conditions (highest priority)
        if next_simulation.game_over:
            if next_simulation.is_win_condition():
                return 10.0  # Major reward for winning
            else:
                return -5.0  # Major penalty for losing
        
        # 2. Progress toward victory: discovering cures
        current_cures = len(getattr(simulation, 'discovered_cures', []))
        next_cures = len(getattr(next_simulation, 'discovered_cures', []))
        
        if next_cures > current_cures:
            reward += 5.0  # Major reward for discovering a cure
        
        # 3. Action-specific rewards
        if action_type == "treat":
            city = action.get("city", player.city)
            
            if city and city.infection_level > 0:
                # Higher reward for treating high infection levels
                infection_level = city.infection_level
                
                if infection_level >= 3:
                    reward += 2.0  # Critical infection
                elif infection_level == 2:
                    reward += 1.0  # Moderate infection
                else:
                    reward += 0.5  # Low infection
                
                # Extra reward if city has infected neighbors (preventing cascade)
                infected_neighbors = sum(1 for neighbor in getattr(city, 'neighbours', [])
                                        if getattr(neighbor, 'infection_level', 0) >= 2)
                if infected_neighbors >= 2:
                    reward += 0.5  # Preventing potential cascade
        
        elif action_type == "build":
            # Building research stations is strategically important
            if not player.city.has_research_station:
                reward += 1.0
                
                # Add additional reward for good distribution (first station or different region)
                research_stations = [city for city in simulation.cities 
                                   if getattr(city, 'has_research_station', False)]
                
                if not research_stations:
                    reward += 0.5  # First station
                else:
                    # Check if this station is in a new region using city name as proxy
                    station_region_markers = [city.name[0] for city in research_stations]
                    if player.city.name[0] not in station_region_markers:
                        reward += 0.5  # Likely new region
        
        elif action_type == "move":
            target_city = action.get("target_city")
            
            if target_city:
                # Moving to treat infection
                if target_city.infection_level >= 2:
                    reward += 0.5
                
                # Moving to research station
                if target_city.has_research_station:
                    reward += 0.3
                
                # Moving to other players for cooperation
                other_players_present = any(p != player and p.city == target_city 
                                         for p in simulation.players)
                if other_players_present:
                    reward += 0.5
        
        elif action_type == "discover_cure":
            # Already covered by cure count check, but add slight bonus for attempt
            reward += 0.2
        
        elif action_type == "share_knowledge":
            # Reward knowledge sharing (cooperative behavior)
            reward += 1.0
            
            # Extra reward if it gets player closer to a cure
            recipient = action.get("recipient")
            if recipient:
                card = action.get("card")
                if card and hasattr(card, 'color'):
                    color = card.color
                    recipient_cards = [c for c in recipient.hand 
                                     if hasattr(c, 'color') and c.color == color]
                    
                    is_scientist = hasattr(recipient, 'role') and recipient.role.name == "Scientist"
                    cards_needed = 4 if is_scientist else 5
                    
                    if len(recipient_cards) + 1 >= cards_needed:
                        reward += 1.0  # Gets them to or close to a cure
        
        # 4. Strategic positioning and preparation
        if action_type == "move":
            target_city = action.get("target_city")
            
            has_matching_card = any(hasattr(card, 'city_name') and card.city_name == target_city.name
                                 for card in player.hand)
            
            if has_matching_card and not target_city.has_research_station:
                reward += 0.2  # Potential future research station
        
        # 5. Penalize for ineffective actions
        if action_type == "pass":
            reward -= 0.3  # Minor penalty for passing
        
        # 6. Global state improvement/deterioration
        current_infection = sum(c.infection_level for c in simulation.cities)
        next_infection = sum(c.infection_level for c in next_simulation.cities)
        
        infection_change = current_infection - next_infection
        reward += infection_change * 0.3  # Scale appropriately
        
        # 7. Outbreak prevention bonus
        current_outbreaks = simulation.outbreak_count
        next_outbreaks = next_simulation.outbreak_count
        
        if current_outbreaks == next_outbreaks:
            # Reward stability (no new outbreaks)
            reward += 0.2
        else:
            # Major penalty for new outbreaks
            reward -= 1.0 * (next_outbreaks - current_outbreaks)
        
        return reward
    
    def optimize_model(self):
        """
        Train the model using a batch from replay memory
        """
        if len(self.memory.memory) < self.batch_size:
            return
        
        # Sample batch with prioritization
        batch, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert batch to tensors
        states = torch.FloatTensor([experience.state for experience in batch]).to(self.device)
        actions_idx = torch.LongTensor([[self._action_to_index(experience.action)] for experience in batch]).to(self.device)
        rewards = torch.FloatTensor([experience.reward for experience in batch]).to(self.device)
        next_states = torch.FloatTensor([experience.next_state for experience in batch]).to(self.device)
        dones = torch.FloatTensor([experience.done for experience in batch]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions_idx)
        
        # Double DQN: Use policy net to select actions, target net to evaluate them
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            
            # Calculate expected Q values
            expected_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones.unsqueeze(1)))
        
        # Calculate loss with importance sampling weights
        loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        weighted_loss = (weights.unsqueeze(1) * loss).mean()
        
        # Update priorities in replay buffer
        td_errors = torch.abs(expected_q_values - current_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update of target network
        self._soft_update_target_network()
    
    def _soft_update_target_network(self):
        """Update target network parameters with a soft update"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + policy_param.data * self.tau)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Add experience to memory and track rewards"""
        self.memory.add(state, action, reward, next_state, done)
        self.episode_rewards += reward
        
        if done:
            self.rewards_history.append(self.episode_rewards)
            self.episode_rewards = 0
    
    def decide_action(self, player, simulation):
        """
        Main decision function that selects the best action based on current state
        """
        # Update game assessment for strategic context
        self._update_team_strategy(simulation)
        
        # Encode the state with strategic information
        state = self.encode_strategic_state(player, simulation)
        
        # Get all possible actions
        possible_actions = self._get_possible_actions(player, simulation)
        
        if not possible_actions:
            return None
        
        # Select action using policy
        selected_action = self.select_action(state, possible_actions)
        
        if not selected_action:
            return None
            
        # Simulate the effect of the action
        next_simulation = self._simulate_action(simulation, player, selected_action)
        next_state = self.encode_strategic_state(player, next_simulation)
        
        # Calculate reward based on strategic goals
        reward = self.calculate_strategic_reward(player, simulation, selected_action, next_simulation)
        
        # Check if this is a terminal state
        done = next_simulation.game_over
        
        # Store the experience for learning
        self.store_experience(state, selected_action, reward, next_state, done)
        
        # Train the model periodically
        self.total_train_steps += 1
        if self.total_train_steps % self.update_frequency == 0:
            self.optimize_model()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Log action for analysis
        self._log_action(selected_action, reward)
        
        return selected_action
    
    def _log_action(self, action, reward):
        """Track action statistics for analysis"""
        action_type = action.get("type", "unknown")
        
        if action_type not in self.action_stats:
            self.action_stats[action_type] = {
                "count": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0
            }
        
        self.action_stats[action_type]["count"] += 1
        self.action_stats[action_type]["total_reward"] += reward
        self.action_stats[action_type]["avg_reward"] = (
            self.action_stats[action_type]["total_reward"] / 
            self.action_stats[action_type]["count"]
        )
    
    def _get_possible_actions(self, player, simulation):
        """Get all valid actions for the current player"""
        actions = []
        
        # MOVE actions (standard movement to adjacent cities)
        for neighbor in player.city.neighbours:
            actions.append({
                "type": "move",
                "target_city": neighbor,
                "method": "standard"
            })
        
        # Direct flight (using city cards)
        for card in player.hand:
            if hasattr(card, 'city_name'):
                target_city = next((city for city in simulation.cities 
                                  if city.name == card.city_name), None)
                if target_city and target_city != player.city:
                    actions.append({
                        "type": "move",
                        "target_city": target_city,
                        "method": "direct_flight",
                        "card": card
                    })
        
        # TREAT action if current city has infection
        if player.city.infection_level > 0:
            actions.append({
                "type": "treat",
                "city": player.city
            })
        
        # BUILD RESEARCH STATION action
        if not player.city.has_research_station:
            city_card = next((card for card in player.hand 
                           if hasattr(card, 'city_name') and card.city_name == player.city.name), None)
            
            # Check if player is Operations Expert or has city card
            is_ops_expert = hasattr(player, 'role') and getattr(player.role, 'name', '') == "Operations_Expert"
            
            if city_card or is_ops_expert:
                actions.append({
                    "type": "build",
                    "city": player.city,
                    "card": city_card
                })
        
        # DISCOVER CURE action
        if player.city.has_research_station:
            # Group cards by color
            cards_by_color = defaultdict(list)
            for card in player.hand:
                if hasattr(card, 'color') and card.color not in ["", "INF"]:
                    cards_by_color[card.color].append(card)
            
            # Check if player can discover any cures
            discovered_cures = getattr(simulation, 'discovered_cures', [])
            
            # Consider role abilities (Scientist needs fewer cards)
            is_scientist = hasattr(player, 'role') and getattr(player.role, 'name', '') == "Scientist"
            cards_needed = 4 if is_scientist else 5
            
            for color, cards in cards_by_color.items():
                if color not in discovered_cures and len(cards) >= cards_needed:
                    actions.append({
                        "type": "discover_cure",
                        "color": color,
                        "cards": cards[:cards_needed]  # Use just enough cards
                    })
        
        # SHARE KNOWLEDGE action
        same_city_players = [p for p in simulation.players if p.id != player.id and p.city == player.city]
        
        if same_city_players:
            # Check if player is researcher (can share any card)
            is_researcher = hasattr(player, 'role') and getattr(player.role, 'name', '') == "Researcher"
            
            # Current city cards for sharing
            sharable_cards = []
            if is_researcher:
                # Researcher can share any city card
                sharable_cards = [card for card in player.hand 
                                if hasattr(card, 'city_name') and card.city_name]
            else:
                # Others can only share current city card
                sharable_cards = [card for card in player.hand 
                                if hasattr(card, 'city_name') and card.city_name == player.city.name]
            
            # Create share actions for each combination
            for card in sharable_cards:
                for other_player in same_city_players:
                    actions.append({
                        "type": "share_knowledge",
                        "card": card,
                        "direction": "give",
                        "recipient": other_player
                    })
        
        # PASS action (always available)
        actions.append({
            "type": "pass"
        })
        
        return actions
    
    def _simulate_action(self, simulation, player, action):
        """Simulate the effect of an action on the game state"""
        from copy import deepcopy
        
        # Create a deep copy of the simulation to avoid modifying the real state
        sim_copy = deepcopy(simulation)
        
        # Find the player in the copied simulation
        player_copy = None
        for p in sim_copy.players:
            if p.id == player.id:
                player_copy = p
                break
        
        if not player_copy:
            return sim_copy  # Cannot find player, return unmodified copy
        
        # Apply the action effects
        action_type = action.get("type")
        
        if action_type == "move":
            target_city = action.get("target_city")
            method = action.get("method", "standard")
            card = action.get("card")
            
            if target_city:
                # Find corresponding city in the simulation copy
                target_city_copy = next((city for city in sim_copy.cities if city.name == target_city.name), None)
                
                if target_city_copy:
                    # Move player
                    player_copy.city = target_city_copy
                    
                    # Handle card discard for certain movement types
                    if method in ["direct_flight", "charter_flight"] and card:
                        matching_card = next((c for c in player_copy.hand 
                                           if hasattr(c, 'city_name') and c.city_name == card.city_name), None)
                        
                        if matching_card:
                            player_copy.hand.remove(matching_card)
                            sim_copy.player_discard_pile.append(matching_card)
        
        elif action_type == "treat":
            city = action.get("city", player_copy.city)
            
            if city and city.infection_level > 0:
                # Find the city in the simulation copy
                city_copy = next((c for c in sim_copy.cities if c.name == city.name), None)
                
                if city_copy:
                    # Check if disease is cured
                    is_cured = any(disease["color"] == "Blue" and disease["cured"] 
                                 for disease in sim_copy.diseases)
                    
                    if is_cured:
                        # Remove all cubes if disease is cured
                        city_copy.infection_level = 0
                    else:
                        # Otherwise remove one cube
                        city_copy.infection_level -= 1
        
        elif action_type == "build":
            # Build research station in current city
            player_copy.city.has_research_station = True
            
            # Discard city card if not Operations Expert
            is_ops_expert = hasattr(player_copy, 'role') and player_copy.role.name == "Operations_Expert"
            
            if not is_ops_expert:
                card = action.get("card")
                if card:
                    matching_card = next((c for c in player_copy.hand 
                                       if hasattr(c, 'city_name') and c.city_name == card.city_name), None)
                    
                    if matching_card:
                        player_copy.hand.remove(matching_card)
                        sim_copy.player_discard_pile.append(matching_card)
        
        elif action_type == "discover_cure":
            color = action.get("color")
            cards = action.get("cards", [])
            
            # Find disease in simulation copy and mark as cured
            for disease in sim_copy.diseases:
                if disease["color"] == color:
                    disease["cured"] = True
                    
                    # Add to discovered cures list
                    if color not in sim_copy.discovered_cures:
                        sim_copy.discovered_cures.append(color)
            
            # Discard used cards
            for card in cards:
                matching_card = next((c for c in player_copy.hand 
                                   if hasattr(c, 'color') and c.color == card.color), None)
                
                if matching_card:
                    player_copy.hand.remove(matching_card)
                    sim_copy.player_discard_pile.append(matching_card)
        
        elif action_type == "share_knowledge":
            card = action.get("card")
            recipient = action.get("recipient")
            direction = action.get("direction", "give")
            
            if card and recipient:
                # Find recipient in simulation copy
                recipient_copy = next((p for p in sim_copy.players if p.id == recipient.id), None)
                
                if recipient_copy:
                    # Find matching card in player's hand
                    matching_card = next((c for c in player_copy.hand 
                                       if hasattr(c, 'city_name') and c.city_name == card.city_name), None)
                    
                    if matching_card and direction == "give":
                        # Transfer card from player to recipient
                        player_copy.hand.remove(matching_card)
                        recipient_copy.hand.append(matching_card)
        
        # Update game over status
        sim_copy.game_over = False
        
        # Check for win condition (all diseases cured)
        all_cured = all(disease["cured"] for disease in sim_copy.diseases)
        if all_cured:
            sim_copy.game_over = True
        
        # Check for outbreak limit
        if sim_copy.outbreak_count >= sim_copy.outbreak_limit:
            sim_copy.game_over = True
        
        return sim_copy
    
    def save_state(self, filepath="marl_agent_state.pt"):
        """Save agent state to file"""
        save_data = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_train_steps": self.total_train_steps,
            "rewards_history": self.rewards_history,
            "action_stats": self.action_stats
        }
        
        torch.save(save_data, filepath)
        print(f"MARL agent state saved in {filepath} (epsilon: {self.epsilon:.4f}, steps: {self.total_train_steps})")
    
    def load_state(self, filepath="marl_agent_state.pt"):
        """Load agent state from file"""
        try:
            if os.path.exists(filepath):
                save_data = torch.load(filepath, map_location=self.device)
                
                self.policy_net.load_state_dict(save_data["policy_net"])
                self.target_net.load_state_dict(save_data["target_net"])
                self.optimizer.load_state_dict(save_data["optimizer"])
                self.epsilon = save_data.get("epsilon", self.epsilon)
                self.total_train_steps = save_data.get("total_train_steps", 0)
                self.rewards_history = save_data.get("rewards_history", [])
                self.action_stats = save_data.get("action_stats", {})
                
                print(f"Loaded MARL state from {filepath}")
                return True
            
            print(f"No saved state found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading MARL state: {e}")
            return False


# Global agent instance for strategy function
_global_marl_agent = None

def marl_agent_strategy(player):
    """
    Strategy function that uses the improved MARL agent for decision making
    """
    global _global_marl_agent
    
    # Initialize agent if needed
    import os
    agent_state_dir = "./agents_state"
    os.makedirs(agent_state_dir, exist_ok=True)
    state_file = os.path.join(agent_state_dir, "marl_agent_state.pt")
    
    if _global_marl_agent is None:
        _global_marl_agent = ImprovedMARLAgent()
        print(f"Created new MARL agent (state file: {state_file})")
        _global_marl_agent.load_state(state_file)
    
    # Enable debug logging
    debug_enabled = True
    
    def debug_log(message):
        if debug_enabled:
            print(f"[MARL-DEBUG] {message}")
    
    # Get simulation from player
    simulation = player.simulation
    
    # Check for emergency situations first
    if player.city.infection_level >= 3:
        debug_log(f"EMERGENCY: Critical infection in {player.city.name}")
        return {"type": "treat", "city": player.city}
    
    # Check for hand limit
    if len(player.hand) > 7:
        # Find least valuable card to discard
        city_cards = [card for card in player.hand if hasattr(card, 'city_name')]
        if city_cards:
            # Prefer to discard city cards that don't match current city
            non_matching_cards = [card for card in city_cards 
                                if card.city_name != player.city.name]
            if non_matching_cards:
                debug_log(f"HAND LIMIT: Discarding card for {non_matching_cards[0].city_name}")
                return {"type": "discard", "card": non_matching_cards[0]}
            debug_log(f"HAND LIMIT: Discarding card for {city_cards[0].city_name}")
            return {"type": "discard", "card": city_cards[0]}
        
        # If no city cards, discard first card
        debug_log(f"HAND LIMIT: Discarding card (no suitable city cards)")
        return {"type": "discard", "card": player.hand[0]}
    
    # Update strategy based on game state
    _global_marl_agent._update_team_strategy(simulation)
    priority = _global_marl_agent.team_strategy.get("priority", "balanced_approach")
    debug_log(f"STRATEGY: {priority}")
    
    # Check for cure discovery opportunity (highest priority if possible)
    if player.city.has_research_station:
        cards_by_color = defaultdict(list)
        for card in player.hand:
            if hasattr(card, 'color') and card.color and card.color != "INF":
                cards_by_color[card.color].append(card)
        
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        is_scientist = hasattr(player, 'role') and getattr(player.role, 'name', '') == "Scientist"
        cards_needed = 4 if is_scientist else 5
        
        for color, cards in cards_by_color.items():
            if color not in discovered_cures and len(cards) >= cards_needed:
                debug_log(f"PRIORITY: Discovering cure for {color}")
                return {
                    "type": "discover_cure",
                    "color": color,
                    "cards": cards[:cards_needed]
                }
    
    # Regular decision making
    action = _global_marl_agent.decide_action(player, simulation)
    
    # Log selected action for debugging
    if action:
        action_type = action.get("type", "unknown")
        if action_type == "move":
            target_city = action.get("target_city")
            if target_city:
                debug_log(f"ACTION: Moving to {target_city.name}")
        elif action_type == "treat":
            debug_log(f"ACTION: Treating infection in {player.city.name}")
        elif action_type == "build":
            debug_log(f"ACTION: Building research station in {player.city.name}")
        elif action_type == "share_knowledge":
            recipient = action.get("recipient")
            card = action.get("card")
            if recipient and card:
                debug_log(f"ACTION: Sharing card for {card.city_name} with {recipient.name}")
        else:
            debug_log(f"ACTION: {action_type}")
    
    # Periodically save agent state
    if random.random() < 0.1:  # 10% chance each turn
        _global_marl_agent.save_state(state_file)
    
    return action