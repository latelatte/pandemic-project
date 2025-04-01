import math
import random
import time
from copy import deepcopy
import numpy as np
import pickle
import os
from collections import defaultdict, Counter

class ImprovedMCTSNode:
    """Enhanced Monte Carlo Tree Search node with strategic evaluation"""
    def __init__(self, state, parent=None, action=None, exploration_weight=1.4, player_id=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None  # Lazy initialization
        self.exploration_weight = exploration_weight
        self.player_id = player_id  # Track which player this node represents
        self.action_stats = {}  # Track statistics for each action
        self.strategic_value = None  # Cache for strategic evaluation
        
    def get_untried_actions(self, action_generator=None):
        """Get untried actions with lazy initialization"""
        if self.untried_actions is None:
            if action_generator:
                self.untried_actions = action_generator(self.state)
            else:
                self.untried_actions = self._default_action_generator()
        return self.untried_actions
        
    def _default_action_generator(self):
        """Default action generator if none is provided"""
        actions = []
        player = self.state.get("current_player")
        
        if player and player.city:
            # Move actions
            for neighbor in player.city.neighbours:
                if neighbor != player.city:
                    actions.append({
                        "type": "move", 
                        "target_city": neighbor
                    })
            
            # Treat action
            if player.city.infection_level > 0:
                actions.append({
                    "type": "treat",
                    "city": player.city
                })
            
            # Build research station
            if not player.city.has_research_station:
                for card in player.hand:
                    if hasattr(card, 'city_name') and card.city_name == player.city.name:
                        actions.append({
                            "type": "build",
                            "city": player.city,
                            "card": card
                        })
                        break
                
            # Discover cure actions
            if player.city.has_research_station:
                cards_by_color = defaultdict(list)
                for card in player.hand:
                    if hasattr(card, 'color') and card.color and card.color != "INF":
                        cards_by_color[card.color].append(card)
                
                # Check for diseases that can be cured
                discovered_cures = self.state.get("discovered_cures", [])
                is_scientist = hasattr(player, 'role') and getattr(player.role, 'name', '') == "Scientist"
                cards_needed = 4 if is_scientist else 5
                
                for color, cards in cards_by_color.items():
                    if color not in discovered_cures and len(cards) >= cards_needed:
                        actions.append({
                            "type": "discover_cure",
                            "color": color,
                            "cards": cards[:cards_needed]
                        })
            
            # Share knowledge actions
            same_city_players = [p for p in self.state.get("players", []) 
                                if p.id != player.id and p.city == player.city]
            
            if same_city_players:
                # Check if player has current city card
                city_card = next((card for card in player.hand 
                                if hasattr(card, 'city_name') and card.city_name == player.city.name), None)
                
                if city_card:
                    for other_player in same_city_players:
                        actions.append({
                            "type": "share_knowledge",
                            "direction": "give",
                            "target_player": other_player,
                            "card": city_card
                        })
            
            # Pass action
            actions.append({
                "type": "pass",
                "base_value": 0.01
            })
        
        return actions
        
    def is_fully_expanded(self, action_generator=None):
        """Check if all possible actions have been tried"""
        return len(self.get_untried_actions(action_generator)) == 0
    
    def is_terminal(self):
        """Check if this is a terminal state (game over)"""
        return self.state.get("game_over", False)
    
    def select_child(self):
        """Select the most promising child node using UCB1"""
        log_visits = math.log(self.visits + 1e-10)
        
        def ucb_score(child):
            # Exploitation term
            exploitation = child.value / (child.visits + 1e-10)
            
            # Exploration term with strategic bias
            exploration = self.exploration_weight * math.sqrt(log_visits / (child.visits + 1e-10))
            
            # Strategic bias based on action type
            strategic_bias = 0.0
            if child.action:
                action_type = child.action.get("type", "")
                if action_type == "discover_cure":
                    strategic_bias = 0.2  # Bonus for cure discovery
                elif action_type == "treat" and child.action.get("city") and child.action.get("city").infection_level >= 3:
                    strategic_bias = 0.1  # Bonus for treating high infection
            
            return exploitation + exploration + strategic_bias
        
        return max(self.children, key=ucb_score)
    
    def expand(self, action_generator=None):
        """Expand the tree by adding a new child node"""
        if self.is_terminal():
            return None
            
        untried_actions = self.get_untried_actions(action_generator)
        if not untried_actions:
            return None
            
        action = untried_actions.pop()
        next_state = self.apply_action(self.state, action)
        
        # Determine next player ID
        next_player_id = self.player_id
        if "current_player_index" in self.state:
            current_idx = self.state["current_player_index"]
            next_idx = (current_idx + 1) % len(self.state.get("players", []))
            next_player_id = self.state["players"][next_idx].id
        
        child = ImprovedMCTSNode(
            next_state, 
            parent=self, 
            action=action, 
            exploration_weight=self.exploration_weight,
            player_id=next_player_id
        )
        
        self.children.append(child)
        return child
        
    def apply_action(self, state, action):
        """Apply an action to the state and return the new state"""
        new_state = deepcopy(state)
        player = new_state.get("current_player")
        
        if not player:
            return new_state
        
        action_type = action.get("type")
        
        if action_type == "move":
            target_city = action.get("target_city")
            if target_city and (target_city.name != player.city.name):
                # Find corresponding city in the state
                for city in new_state.get("cities", []):
                    if city.name == target_city.name:
                        player.city = city
                        break
        
        elif action_type == "treat":
            city = action.get("city") or player.city
            if city and city.infection_level > 0:
                # Check if disease is cured
                disease_cured = False
                
                # Use city name first character as proxy for disease color
                first_char = city.name[0].upper()
                disease_color = "Blue"  # Default
                
                if 'A' <= first_char <= 'G':
                    disease_color = "Blue"
                elif 'H' <= first_char <= 'M':
                    disease_color = "Yellow"
                elif 'N' <= first_char <= 'S':
                    disease_color = "Black"
                else:  # T-Z
                    disease_color = "Red"
                
                # Check if this disease is cured
                for disease in new_state.get("diseases", []):
                    if disease.get("color") == disease_color and disease.get("cured", False):
                        disease_cured = True
                        break
                
                if disease_cured:
                    # Remove all cubes if disease is cured
                    city.infection_level = 0
                else:
                    # Otherwise remove one cube
                    city.infection_level = max(0, city.infection_level - 1)
        
        elif action_type == "build":
            if player.city and not player.city.has_research_station:
                player.city.has_research_station = True
                
                # Use card if not Operations Expert
                is_ops_expert = hasattr(player, 'role') and player.role.name == "Operations Expert"
                if not is_ops_expert:
                    card = action.get("card")
                    if card:
                        # Remove card from hand
                        for i, c in enumerate(player.hand):
                            if hasattr(c, 'city_name') and c.city_name == card.city_name:
                                player.hand.pop(i)
                                break
                
        elif action_type == "discover_cure":
            color = action.get("color")
            cards = action.get("cards", [])
            
            # Remove cards from hand
            card_indices = []
            for card in cards:
                for i, c in enumerate(player.hand):
                    if hasattr(c, 'color') and c.color == card.color:
                        card_indices.append(i)
                        break
            
            # Remove in reverse order to maintain correct indices
            for i in sorted(card_indices, reverse=True):
                if i < len(player.hand):
                    player.hand.pop(i)
            
            # Mark disease as cured
            for disease in new_state.get("diseases", []):
                if disease.get("color") == color:
                    disease["cured"] = True
                    break
            
            # Add to discovered cures list
            if "discovered_cures" not in new_state:
                new_state["discovered_cures"] = []
            if color not in new_state["discovered_cures"]:
                new_state["discovered_cures"].append(color)
        
        elif action_type == "share_knowledge":
            card = action.get("card")
            target_player = action.get("target_player")
            direction = action.get("direction", "give")
            
            if card and target_player:
                # Find target player in new state
                for p in new_state.get("players", []):
                    if p.id == target_player.id:
                        target_player = p
                        break
                
                # Find card in player's hand
                card_index = -1
                for i, c in enumerate(player.hand):
                    if hasattr(c, 'city_name') and c.city_name == card.city_name:
                        card_index = i
                        break
                
                if card_index >= 0:
                    if direction == "give":
                        # Transfer card from player to target
                        transferred_card = player.hand.pop(card_index)
                        target_player.hand.append(transferred_card)
                    else:  # Take
                        # Implementation for "take" direction would go here
                        pass
        
        # Update game state flags
        
        # Check for win condition (all diseases cured)
        all_cured = True
        for disease in new_state.get("diseases", []):
            if not disease.get("cured", False):
                all_cured = False
                break
                
        if all_cured:
            new_state["game_over"] = True
            new_state["win"] = True
        
        # Check for outbreak limit exceeded
        outbreak_count = new_state.get("outbreak_count", 0)
        outbreak_limit = new_state.get("outbreak_limit", 8)
        if outbreak_count >= outbreak_limit:
            new_state["game_over"] = True
            new_state["win"] = False
        
        return new_state
    
    def update(self, result):
        """Update node statistics with simulation result"""
        self.visits += 1
        self.value += result
        
        # Update action statistics
        if self.action:
            action_type = self.action.get("type", "unknown")
            if action_type not in self.action_stats:
                self.action_stats[action_type] = {
                    "visits": 0,
                    "value": 0.0
                }
            
            self.action_stats[action_type]["visits"] += 1
            self.action_stats[action_type]["value"] += result
    
    def get_strategic_value(self):
        """Get cached strategic value or compute it"""
        if self.strategic_value is None:
            self.strategic_value = self._evaluate_strategic_value()
        return self.strategic_value
    
    def _evaluate_strategic_value(self):
        """Evaluate the strategic value of the state"""
        state = self.state
        
        # If terminal state, return direct outcome
        if state.get("game_over", False):
            return 1.0 if state.get("win", False) else 0.0
        
        # Start with base score
        score = 0.5
        
        # 1. Progress toward victory (cures discovered)
        discovered_cures = state.get("discovered_cures", [])
        cure_score = len(discovered_cures) / 4.0  # 4 cures needed for victory
        score += 0.25 * cure_score
        
        # 2. Infection control
        total_infection = 0
        max_possible_infection = 0
        high_infection_count = 0
        
        for city in state.get("cities", []):
            infection_level = getattr(city, 'infection_level', 0)
            total_infection += infection_level
            max_possible_infection += 3  # Assuming max infection level is 3
            
            if infection_level >= 3:
                high_infection_count += 1
        
        infection_control_score = 1.0 - (total_infection / max_possible_infection) if max_possible_infection > 0 else 1.0
        infection_control_score -= 0.1 * high_infection_count  # Penalty for high infection cities
        
        score += 0.25 * infection_control_score
        
        # 3. Research station coverage
        research_stations = [city for city in state.get("cities", []) if getattr(city, 'has_research_station', False)]
        station_score = min(1.0, len(research_stations) / 5.0)  # Optimal number is about 5
        
        # Bonus for good distribution using first letter as region proxy
        station_regions = set(city.name[0] for city in research_stations)
        region_coverage = len(station_regions) / 4.0  # Ideally cover 4 regions
        
        score += 0.15 * (0.5 * station_score + 0.5 * region_coverage)
        
        # 4. Outbreak risk
        outbreak_count = state.get("outbreak_count", 0)
        outbreak_limit = state.get("outbreak_limit", 8)
        outbreak_safety = 1.0 - (outbreak_count / outbreak_limit)
        
        score += 0.1 * outbreak_safety
        
        # 5. Team positioning and preparation
        cities_with_players = set()
        player_near_research = 0
        
        for player in state.get("players", []):
            if player.city:
                cities_with_players.add(player.city.name)
                
                if player.city.has_research_station:
                    player_near_research += 1
        
        # Coverage score (players spread out vs clustered)
        coverage_score = min(1.0, len(cities_with_players) / len(state.get("players", [])))
        
        # Research access score
        research_access = player_near_research / len(state.get("players", []))
        
        score += 0.1 * (0.5 * coverage_score + 0.5 * research_access)
        
        # 6. Card distribution for cures
        cards_by_color = defaultdict(int)
        
        for player in state.get("players", []):
            for card in player.hand:
                if hasattr(card, 'color') and card.color and card.color != "INF":
                    cards_by_color[card.color] += 1
        
        # For each color, evaluate how close we are to cure
        cards_needed = 5  # Default, simplified for evaluation
        card_preparation = 0.0
        
        for color, count in cards_by_color.items():
            if color not in discovered_cures:
                progress = min(1.0, count / cards_needed)
                card_preparation += progress
        
        # Normalize by undiscovered diseases
        undiscovered = 4 - len(discovered_cures)
        if undiscovered > 0:
            card_preparation /= undiscovered
            score += 0.15 * card_preparation
        
        return max(0.0, min(1.0, score))
    
    def rollout_policy(self, state, depth=0, max_depth=10, visited_states=None):
        """Enhanced rollout policy with strategic heuristics"""
        if visited_states is None:
            visited_states = set()
            
        state_hash = hash(str(state))
        if state_hash in visited_states:
            return self._evaluate_strategic_value() * 0.7
        
        visited_states.add(state_hash)
        
        if depth >= max_depth or state.get("game_over", False):
            return self._evaluate_strategic_value()
            
        player = state.get("current_player")
        
        if not player or not player.city:
            return self._evaluate_strategic_value()
            
        # Generate candidate actions with heuristic weights
        weighted_actions = []
        
        # 1. Treat high infection cities (high priority)
        if player.city.infection_level >= 3:
            weighted_actions.append(({
                "type": "treat",
                "city": player.city
            }, 5.0))  # High weight
        elif player.city.infection_level > 0:
            weighted_actions.append(({
                "type": "treat",
                "city": player.city
            }, 3.0))  # Medium weight
        
        # 2. Move to nearby high infection cities
        for neighbor in player.city.neighbours:
            if neighbor.infection_level >= 3:
                weighted_actions.append(({
                    "type": "move",
                    "target_city": neighbor
                }, 4.0))  # High weight
            elif neighbor.infection_level > 0:
                weighted_actions.append(({
                    "type": "move",
                    "target_city": neighbor
                }, 2.0))  # Medium weight
            else:
                weighted_actions.append(({
                    "type": "move",
                    "target_city": neighbor
                }, 1.0))  # Low weight
        
        # 3. Build research station if none nearby
        research_stations = [city for city in state.get("cities", []) if getattr(city, 'has_research_station', False)]
        research_nearby = any(rs in player.city.neighbours for rs in research_stations)
        
        if not player.city.has_research_station and not research_nearby:
            city_card = next((card for card in player.hand 
                           if hasattr(card, 'city_name') and card.city_name == player.city.name), None)
            
            if city_card:
                weighted_actions.append(({
                    "type": "build",
                    "city": player.city,
                    "card": city_card
                }, 3.5))  # High-medium weight
        
        # 4. Discover cure if possible
        if player.city.has_research_station:
            # Group cards by color
            cards_by_color = defaultdict(list)
            for card in player.hand:
                if hasattr(card, 'color') and card.color and card.color != "INF":
                    cards_by_color[card.color].append(card)
            
            # Check for diseases that can be cured
            discovered_cures = state.get("discovered_cures", [])
            is_scientist = hasattr(player, 'role') and getattr(player.role, 'name', '') == "Scientist"
            cards_needed = 4 if is_scientist else 5
            
            for color, cards in cards_by_color.items():
                if color not in discovered_cures and len(cards) >= cards_needed:
                    weighted_actions.append(({
                        "type": "discover_cure",
                        "color": color,
                        "cards": cards[:cards_needed]
                    }, 6.0))  # Highest weight
        
        # 5. Move toward research station if we have cards for cure
        if not player.city.has_research_station:
            # Count cards by color
            cards_by_color = defaultdict(int)
            for card in player.hand:
                if hasattr(card, 'color') and card.color and card.color != "INF":
                    cards_by_color[card.color] += 1
            
            # Check if we're close to having a cure
            discovered_cures = state.get("discovered_cures", [])
            is_scientist = hasattr(player, 'role') and getattr(player.role, 'name', '') == "Scientist"
            cards_needed = 4 if is_scientist else 5
            
            for color, count in cards_by_color.items():
                if color not in discovered_cures and count >= cards_needed - 1:
                    # Find nearby research stations to move toward
                    research_stations = [city for city in state.get("cities", []) 
                                      if getattr(city, 'has_research_station', False)]
                    
                    if research_stations:
                        # Simple heuristic: move toward any adjacent city
                        for neighbor in player.city.neighbours:
                            weighted_actions.append(({
                                "type": "move",
                                "target_city": neighbor
                            }, 2.5))  # Medium-high weight
        
        # If no weighted actions, add pass with low weight
        if not weighted_actions:
            weighted_actions.append(({
                "type": "pass"
            }, 0.5))  # Low weight
        
        # Select action based on weights
        total_weight = sum(weight for _, weight in weighted_actions)
        if total_weight <= 0:
            action = {"type": "pass"}
        else:
            selection = random.uniform(0, total_weight)
            cumulative_weight = 0
            
            for action, weight in weighted_actions:
                cumulative_weight += weight
                if selection <= cumulative_weight:
                    break
            
        # Apply selected action and continue rollout
        next_state = self.apply_action(state, action)
        
        # Continue rollout with next player
        next_state["current_player"] = self._get_next_player(state)
        
        return self.rollout_policy(next_state, depth + 1, max_depth)
    
    def _get_next_player(self, state):
        """Get the next player in the simulation"""
        players = state.get("players", [])
        if not players:
            return None
            
        current_player = state.get("current_player")
        if not current_player:
            return players[0]
            
        # Find current player index
        current_idx = -1
        for i, player in enumerate(players):
            if player.id == current_player.id:
                current_idx = i
                break
        
        # Get next player
        next_idx = (current_idx + 1) % len(players)
        return players[next_idx]


class ImprovedMCTSAgent:
    """
    Enhanced Monte Carlo Tree Search agent for Pandemic with strategic heuristics
    and collaboration awareness.
    """
    def __init__(self, name="MCTS", simulation_count=80, exploration_weight=1.4,
                max_rollout_depth=7, time_limit=0.5, use_heuristics=True):
        self.name = name
        self.simulation_count = simulation_count
        self.exploration_weight = exploration_weight
        self.max_rollout_depth = max_rollout_depth
        self.max_time = time_limit
        self.use_heuristics = use_heuristics
        
        self.node_cache = {}
        self.action_stats = {}
        self.visit_counts = {}
        self.value_sum = {}
        
        self.infection_hotspots = []
        self.team_strategy = {}
        self.cure_progress = defaultdict(float)
        self.movement_history = {}
        
        self.last_simulation_count = 0
        self.thinking_time = 0
        self.total_simulations = 0
    
    def _extract_state(self, simulation, player):
        """Extract relevant state information from the game simulation"""
        state = {
            "current_player": player,
            "cities": simulation.cities,
            "players": simulation.players,
            "diseases": getattr(simulation, "diseases", []),
            "discovered_cures": getattr(simulation, "discovered_cures", []),
            "outbreak_count": simulation.outbreak_count,
            "outbreak_limit": simulation.outbreak_limit,
            "game_over": simulation.game_over,
            "win": False,
            "current_player_index": simulation.players.index(player) if player in simulation.players else 0
        }
        return state
    
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
        
        # Update cure progress tracking
        colors = ["Blue", "Red", "Yellow", "Black"]
        for color in colors:
            if color in discovered_cures:
                self.cure_progress[color] = 1.0
            else:
                # Count cards of this color across all players
                total_cards = sum(1 for player in simulation.players
                               for card in player.hand
                               if hasattr(card, 'color') and card.color == color)
                
                # Approximate progress (5 cards needed, scientist needs 4)
                scientist_present = any(getattr(player, 'role', None) and 
                                      getattr(player.role, 'name', '') == "Scientist"
                                      for player in simulation.players)
                
                cards_needed = 4 if scientist_present else 5
                self.cure_progress[color] = min(0.99, total_cards / (cards_needed * 1.5))
        
        # Identify infection hotspots
        self.infection_hotspots = sorted(
            [city for city in simulation.cities if city.infection_level >= 2],
            key=lambda city: city.infection_level,
            reverse=True
        )
        
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
    
    def _apply_heuristic_bias(self, node, action):
        """Apply strategic bias to action selection based on heuristics"""
        if not self.use_heuristics:
            return 0.0
            
        bias = 0.0
        action_type = action.get("type", "")
        
        # Strategy-based biases
        priority = self.team_strategy.get("priority", "balanced_approach")
        
        # Cure discovery is generally high priority
        if action_type == "discover_cure":
            bias += 0.3
            
            # Even higher in late game with "final_cure" priority
            if priority == "final_cure":
                bias += 0.2
        
        # Treat infection based on urgency
        elif action_type == "treat":
            city = action.get("city")
            if city:
                # Critical infection (level 3) is high priority
                if city.infection_level >= 3:
                    bias += 0.3
                    
                    # Even higher with "control_outbreaks" priority
                    if priority == "control_outbreaks" or priority == "control_critical":
                        bias += 0.2
                elif city.infection_level == 2:
                    bias += 0.2
                else:
                    bias += 0.1
        
        # Building research stations
        elif action_type == "build":
            # Higher priority early game
            if priority == "research_infrastructure":
                bias += 0.25
            else:
                bias += 0.15
                
            # Check if this is the first research station
            research_stations = [city for city in node.state.get("cities", []) 
                               if getattr(city, 'has_research_station', False)]
            if not research_stations:
                bias += 0.2  # First station is very valuable
                
        # Movement based on target
        elif action_type == "move":
            target_city = action.get("target_city")
            player = node.state.get("current_player")
            if target_city and player and player.city and target_city.name == player.city.name:
                return -10.0
            
            if target_city and player and player.city:
                if target_city.name == player.city.name:
                    return -10.0
                
                if hasattr(player, 'id'):
                    player_id = player.id
                    if player_id in self.movement_history:
                        recent_visits = self.movement_history[player_id]
                        if target_city.name in recent_visits:
                            visits = recent_visits.count(target_city.name)
                            recency_penalty = sum(0.8 for i, city in enumerate(reversed(recent_visits))
                                            if city == target_city.name and i < 3)
                            bias -= (visits * 1.2 + recency_penalty)
            
            if target_city:
                # Moving to high infection city
                if target_city.infection_level >= 3:
                    bias += 0.25
                elif target_city.infection_level >= 2:
                    bias += 0.15
                    
                # Moving to research station
                if target_city.has_research_station:
                    bias += 0.15
                    
                    # Check if we have cards for cure
                    cards_by_color = defaultdict(int)
                    for card in node.state.get("current_player").hand:
                        if hasattr(card, 'color') and card.color and card.color != "INF":
                            cards_by_color[card.color] += 1
                    
                    discovered_cures = node.state.get("discovered_cures", [])
                    is_scientist = hasattr(node.state.get("current_player"), 'role') and getattr(node.state.get("current_player").role, 'name', '') == "Scientist"
                    cards_needed = 4 if is_scientist else 5
                    
                    # Moving to research station with cure potential
                    for color, count in cards_by_color.items():
                        if color not in discovered_cures and count >= cards_needed - 1:
                            bias += 0.3
                
                # Moving to join another player (cooperation)
                other_players = [p for p in node.state.get("players", []) 
                               if p.id != node.state.get("current_player").id and p.city == target_city]
                if other_players:
                    bias += 0.15
        
        # Knowledge sharing
        elif action_type == "share_knowledge":
            bias += 0.2  # Generally useful cooperative action
            
            # Check if this helps complete a set
            card = action.get("card")
            recipient = action.get("recipient")
            
            if card and recipient and hasattr(card, 'color'):
                # Count recipient's cards of this color
                color = card.color
                color_count = sum(1 for c in recipient.hand 
                               if hasattr(c, 'color') and c.color == color)
                
                # Higher bias if this gets close to a cure
                discovered_cures = node.state.get("discovered_cures", [])
                is_scientist = hasattr(recipient, 'role') and getattr(recipient.role, 'name', '') == "Scientist"
                cards_needed = 4 if is_scientist else 5
                
                if color not in discovered_cures:
                    if color_count + 1 >= cards_needed:
                        bias += 0.4  # Completes a set
                    elif color_count + 1 >= cards_needed - 1:
                        bias += 0.3  # Gets very close
        
        # Pass is generally poor
        elif action_type == "pass":
            bias -= 0.1
            
        player = node.state.get("current_player")
        if hasattr(player, 'role') and player.role:
            role_name = getattr(player.role, 'name', '')
            
            # Medic: Prioritize treatment and movement to infection areas
            if role_name == "Medic":
                if action_type == "treat":
                    bias += 0.3  # Medic is extremely efficient at treating
                    
                    # Check if disease is cured (auto-removal capability)
                    city = action.get("city")
                    if city and hasattr(city, 'color'):
                        color = city.color
                        if color in node.state.get("discovered_cures", []):
                            bias += 0.2  # Even more valuable with cured disease
                
                elif action_type == "move":
                    # Medic should prioritize moving to infected cities
                    target_city = action.get("target_city")
                    if target_city and getattr(target_city, 'infection_level', 0) > 0:
                        bias += 0.2 * target_city.infection_level
            
            # Scientist: Prioritize cure discovery and collecting cards
            elif role_name == "Scientist":
                if action_type == "discover_cure":
                    bias += 0.3  # Already good at this
                    
                elif action_type == "move" and priority == "cure_progress":
                    # Moving toward research stations when we have cards
                    target_city = action.get("target_city")
                    if target_city and target_city.has_research_station:
                        # Count cards by color
                        cards_by_color = defaultdict(int)
                        for card in player.hand:
                            if hasattr(card, 'color') and card.color != "INF":
                                cards_by_color[card.color] += 1
                        
                        # If close to cure with 3+ cards of a color
                        if any(count >= 3 for count in cards_by_color.values()):
                            bias += 0.25
            
            # Operations Expert: Prioritize building research stations
            elif role_name == "Operations Expert":
                if action_type == "build":
                    bias += 0.3  # General bonus for building
                    
                    # Check if this is a strategic location
                    city = action.get("city")
                    if city:
                        # Check if this city connects multiple regions
                        if len(getattr(city, 'neighbours', [])) >= 4:
                            bias += 0.2  # Good hub for research station
            
            # Researcher: Prioritize knowledge sharing
            elif role_name == "Researcher":
                if action_type == "share_knowledge":
                    bias += 0.4  # Major bonus for sharing as researcher
                    
                elif action_type == "move":
                    # Prioritize moving to cities with other players
                    target_city = action.get("target_city")
                    if target_city:
                        other_players = [p for p in node.state.get("players", []) 
                                    if p.id != player.id and p.city == target_city]
                        if other_players:
                            bias += 0.25  # Bonus for moving to join others
            
            # Quarantine Specialist: Protect high-risk areas
            elif role_name == "Quarantine Specialist":
                if action_type == "move":
                    target_city = action.get("target_city")
                    if target_city:
                        # Calculate nearby infection
                        nearby_infection = getattr(target_city, 'infection_level', 0)
                        for neighbor in getattr(target_city, 'neighbours', []):
                            nearby_infection += getattr(neighbor, 'infection_level', 0)
                            
                        # Bonus based on nearby infection levels
                        bias += 0.1 * nearby_infection
            
            # Dispatcher: Coordinate team movements
            elif role_name == "Dispatcher":
                if action_type == "move_player":
                    bias += 0.3  # Base bonus for special ability
                    
                    target = action.get("target_player")
                    destination = action.get("target_city")
                    if target and destination:
                        # Moving players toward research stations
                        if getattr(destination, 'has_research_station', False):
                            bias += 0.2
                            
                        # Moving players toward high infection
                        if getattr(destination, 'infection_level', 0) >= 2:
                            bias += 0.25
        
        return bias
    


    
    def _simulate(self, node, depth=0):
        """
        Run a simulation from the given node, using the rollout policy
        """
        if node.is_terminal():
            return 1.0 if node.state.get("win", False) else 0.0
        
        if depth >= self.max_rollout_depth:
            return node.get_strategic_value()
        
        # Use the node's rollout policy
        return node.rollout_policy(node.state, depth, self.max_rollout_depth)
    
    def decide_action(self, player, simulation):
        """
        Perform MCTS to find the best action
        """
        debug_enabled = False
        def debug_log(message):
            if debug_enabled:
                print(f"[MCTS-DEBUG] {message}")
        
        start_time = time.time()
        
        # Check for emergency situations first
        if player.city.infection_level >= 3:
            # debug_log(f"EMERGENCY: Critical infection in {player.city.name}")
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
                    # debug_log(f"HAND LIMIT: Discarding card for {non_matching_cards[0].city_name}")
                    return {"type": "discard", "card": non_matching_cards[0]}
                # debug_log(f"HAND LIMIT: Discarding card for {city_cards[0].city_name}")
                return {"type": "discard", "card": city_cards[0]}
            
            # If no city cards, discard first card
            # debug_log(f"HAND LIMIT: Discarding card (no suitable city cards)")
            return {"type": "discard", "card": player.hand[0]}
        
        # Update team strategy
        self._update_team_strategy(simulation)
        priority = self.team_strategy.get("priority", "balanced_approach")
        # debug_log(f"STRATEGY: {priority}")
        
        # Extract state from simulation
        root_state = self._extract_state(simulation, player)
        
        # Create root node
        root = ImprovedMCTSNode(
            root_state, 
            exploration_weight=self.exploration_weight,
            player_id=player.id
        )
        
        # Check for immediate cure discovery opportunity
        if player.city.has_research_station:
            # Group cards by color
            cards_by_color = defaultdict(list)
            for card in player.hand:
                if hasattr(card, 'color') and card.color and card.color != "INF":
                    cards_by_color[card.color].append(card)
            
            # Check for diseases that can be cured
            discovered_cures = getattr(simulation, 'discovered_cures', [])
            is_scientist = hasattr(player, 'role') and getattr(player.role, 'name', '') == "Scientist"
            cards_needed = 4 if is_scientist else 5
            
            for color, cards in cards_by_color.items():
                if color not in discovered_cures and len(cards) >= cards_needed:
                    # debug_log(f"OPPORTUNITY: Can discover cure for {color}")
                    return {
                        "type": "discover_cure",
                        "color": color,
                        "cards": cards[:cards_needed]
                    }
        
        # MCTS main loop
        simulation_count = 0
        
        while (time.time() - start_time < self.max_time and 
               simulation_count < self.simulation_count):
            
            # Selection phase: traverse the tree to select a node for expansion
            node = root
            
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child()
            
            # Expansion phase: add a new child node if possible
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
                
                if node is None:
                    continue  # No valid expansion, try again
            
            # Simulation phase: run a playout from the expanded node
            result = self._simulate(node)
            
            # Backpropagation phase: update statistics up the tree
            while node is not None:
                node.update(result)
                node = node.parent
            
            simulation_count += 1
        
        self.last_simulation_count = simulation_count
        self.total_simulations += simulation_count
        self.thinking_time = time.time() - start_time
        
        # debug_log(f"Performed {simulation_count} simulations in {self.thinking_time:.3f}s")
        
        # Choose best action (most visited child)
        if not root.children:
            # debug_log("No valid actions found")
            return {"type": "pass"}
        
        # Calculate action scores
        action_scores = []
        for child in root.children:
            # Traditional MCTS score (visits)
            visits_score = child.visits
            
            # Value score (win rate)
            value_score = child.value / max(1, child.visits)
            
            # Apply heuristic bias
            heuristic_bias = self._apply_heuristic_bias(root, child.action)
            
            # Combined score
            combined_score = visits_score * (value_score + heuristic_bias)
            
            action_scores.append((child, combined_score))
        
        # Sort by combined score
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select best action
        best_child, best_score = action_scores[0]
        best_action = best_child.action
        
        # Log top actions for analysis
        if debug_enabled:
            debug_log(f"Top actions:")
            for i, (child, score) in enumerate(action_scores[:3]):
                if i < 3:  # Show top 3
                    action_type = child.action.get("type", "unknown")
                    value = child.value / max(1, child.visits)
                    debug_log(f"  {i+1}. {action_type}: score={score:.2f}, visits={child.visits}, value={value:.3f}")
        
        # Update action statistics
        action_type = best_action.get("type", "unknown")
        state_key = str(hash(frozenset((k, str(v)) for k, v in root_state.items() if k in ["outbreak_count", "current_player_index"])))
        
        if state_key not in self.action_stats:
            self.action_stats[state_key] = {}
            self.visit_counts[state_key] = {}
            self.value_sum[state_key] = {}
        
        if action_type not in self.action_stats[state_key]:
            self.action_stats[state_key][action_type] = 0
            self.visit_counts[state_key][action_type] = 0
            self.value_sum[state_key][action_type] = 0.0
        
        self.visit_counts[state_key][action_type] += best_child.visits
        self.value_sum[state_key][action_type] += best_child.value
        
        if self.visit_counts[state_key][action_type] > 0:
            self.action_stats[state_key][action_type] = (
                self.value_sum[state_key][action_type] / self.visit_counts[state_key][action_type]
            )
        
        # Log decision
        if action_type == "move":
            target_city = best_action.get("target_city")
            if target_city:
                debug_log(f"ACTION: Moving to {target_city.name}")
        elif action_type == "treat":
            debug_log(f"ACTION: Treating infection in {player.city.name}")
        elif action_type == "build":
            debug_log(f"ACTION: Building research station in {player.city.name}")
        elif action_type == "discover_cure":
            color = best_action.get("color")
            debug_log(f"ACTION: Discovering cure for {color}")
        elif action_type == "share_knowledge":
            card = best_action.get("card")
            recipient = best_action.get("recipient") 
            if card and recipient:
                debug_log(f"ACTION: Sharing {card.city_name} card with {recipient.name}")
        else:
            debug_log(f"ACTION: {action_type}")
        # print("===== MCTS ACTION DEBUG =====")
        # print(f"Player: {player.name} at {player.city.name if player.city else 'None'}")
        # if best_action.get("type") == "move":
        #     target_city = best_action.get("target_city")
        #     if target_city and player.city and target_city.name == player.city.name:
        #         print(f"DEBUG: Selected move to: {target_city.name}")
        #         print(f"DEBUG: Current city: {player.city.name}")
        #         print(f"DEBUG: Same city check: {target_city.name == player.city.name}")
                
        #         if target_city.name == player.city.name:
        #             print(f"Warning: MCTS tried moving to same city {player.city.name}, finding alternative action")
                    
        #             # インデントを修正し、代替アクションを実行
        #             if player.city.infection_level > 0:
        #                 best_action = {"type": "treat", "city": player.city}
        #                 print(f"Choosing to treat infection in {player.city.name} instead")
        #             elif player.city.neighbours:
        #                 target = random.choice(player.city.neighbours)
        #                 best_action = {"type": "move", "target_city": target}
        #                 print(f"Choosing to move to {target.name} instead")
        #             else:
        #                 print("Choosing to pass the turn")
        #                 best_action = {"type": "pass"}
        # print(f"MCTS final action: {best_action.get('type')} to {best_action.get('target_city').name if best_action.get('type') == 'move' and best_action.get('target_city') else ''}")
        
        if best_action.get("type") == "move":
            target_city = best_action.get("target_city")
            if target_city:
                player_id = player.id
                if player_id not in self.movement_history:
                    self.movement_history[player_id] = []
                
                self.movement_history[player_id].append(target_city.name)
                
                if len(self.movement_history[player_id]) > 8:
                    self.movement_history[player_id] = self.movement_history[player_id][-8:]
        
        return best_action
    
    def save_state(self, filepath="mctsagent_state.pkl"):
        """Save agent state to file"""
        # Don't save node cache (too large)
        save_data = {
            "action_stats": self.action_stats,
            "visit_counts": self.visit_counts,
            "value_sum": self.value_sum,
            "exploration_weight": self.exploration_weight,
            "last_simulation_count": self.last_simulation_count,
            "thinking_time": self.thinking_time,
            "total_simulations": self.total_simulations,
            "team_strategy": self.team_strategy,
            "cure_progress": dict(self.cure_progress),
            "movement_history": self.movement_history
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        print(f"MCTS agent state saved in {filepath}")

    def load_state(self, filepath="mctsagent_state.pkl"):
        """Load agent state from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    save_data = pickle.load(f)
                
                self.action_stats = save_data.get("action_stats", {})
                self.visit_counts = save_data.get("visit_counts", {})
                self.value_sum = save_data.get("value_sum", {})
                self.exploration_weight = save_data.get("exploration_weight", self.exploration_weight)
                self.last_simulation_count = save_data.get("last_simulation_count", 0)
                self.thinking_time = save_data.get("thinking_time", 0)
                self.total_simulations = save_data.get("total_simulations", 0)
                self.team_strategy = save_data.get("team_strategy", {})
                loaded_cure_progress = save_data.get("cure_progress", {})
                self.movement_history = save_data.get("movement_history", {})
                
                # Convert dictionary to defaultdict
                self.cure_progress = defaultdict(float)
                for color, progress in loaded_cure_progress.items():
                    self.cure_progress[color] = progress
                
                print(f"Loaded MCTS agent state from {filepath}")
                return True
            
            return False
        except Exception as e:
            print(f"Error loading MCTS state: {e}")
            return False


# Global agent instance for strategy function
_global_mcts_agent = None

def mcts_agent_strategy(player):
    """Strategy function that uses the improved MCTS agent for decision making"""
    global _global_mcts_agent
    
    # Initialize agent if needed
    import os
    agent_state_dir = "./agents_state"
    os.makedirs(agent_state_dir, exist_ok=True)
    state_file = os.path.join(agent_state_dir, "mctsagent_state.pkl")
    
    if _global_mcts_agent is None:
        _global_mcts_agent = ImprovedMCTSAgent()
        print(f"Created new MCTS agent (state file: {state_file})")
        _global_mcts_agent.load_state(state_file)
    
    # Get simulation from player
    simulation = player.simulation
    
    # Make decision
    action = _global_mcts_agent.decide_action(player, simulation)
    
    # Periodically save agent state
    if random.random() < 0.1:  # 10% chance each turn
        _global_mcts_agent.save_state(state_file)
    
    return action