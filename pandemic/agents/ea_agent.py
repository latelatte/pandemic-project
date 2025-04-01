import random
import numpy as np
from deap import base, creator, tools, algorithms
import os
import pickle
from collections import defaultdict, Counter

class EAAgent:
    """
    Enhanced Evolutionary Algorithm-based agent for Pandemic game.
    Features  cooperation strategies, long-term planning, and intelligent resource management.
    """
    def __init__(self, name="EA", population_size=50, generations=10, 
                genome_length=50, crossover_rate=0.7, mutation_rate=0.2,
                dynamic_params=True):
        self.name = name
        # Evolutionary algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.genome_length = genome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.dynamic_params = dynamic_params
        
        # Evolutionary computation state
        self.best_population = []
        self.best_individual = None
        self.generation_count = 0
        
        # Game state tracking
        self.previous_cities = {}
        self.movement_history = {}
        self.cure_progress = defaultdict(int)
        self.team_strategy = {}
        self.game_risk_assessment = {}
        self.infection_hotspots = []
        
        # Performance metrics
        self.fitness_history = []
        self.success_counter = {}
        
        # Set up DEAP library
        self._setup_deap()
        
        # Strategy cache
        self._action_eval_cache = {}
        self._chain_eval_cache = {}
        
    def _setup_deap(self):
        """Initialize the evolutionary algorithm configuration"""
        if 'FitnessMax' in creator.__dict__:
            del creator.FitnessMax
        if 'Individual' in creator.__dict__:
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, 
                             creator.Individual, self.toolbox.attr_float, self.genome_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Changed to blend crossover
        self.toolbox.register("mutate", self._adaptive_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _adaptive_mutation(self, individual):
        """Adaptively adjust mutation rate based on generation and performance"""
        # Calculate dynamic mutation rate based on generation progress and fitness trend
        if self.dynamic_params and self.fitness_history:
            fitness_trend = 0
            if len(self.fitness_history) >= 3:
                recent = self.fitness_history[-3:]
                fitness_trend = (recent[-1] - recent[0]) / 3
            
            # Adjust mutation rate: increase when progress stagnates, decrease when improving rapidly
            if fitness_trend < 0.01:  # Stagnating progress
                current_rate = min(0.5, self.mutation_rate * 1.5)
            elif fitness_trend > 0.05:  # Rapid improvement
                current_rate = max(0.05, self.mutation_rate * 0.8)
            else:
                current_rate = self.mutation_rate
        else:
            current_rate = self.mutation_rate
            
        # Apply mutation using adaptive Gaussian mutation
        for i in range(len(individual)):
            if random.random() < current_rate:
                # Use smaller sigma for fine-tuning as evolution progresses
                sigma = max(0.1, 0.3 * (1 - min(1.0, self.generation_count / 100)))
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(0.0, min(1.0, individual[i]))
                
        return individual,
    
    def evaluate_fitness(self, individual, player, simulation):
        """
        Comprehensive evaluation of simulation state fitness
        
        Returns:
            tuple: (fitness_score,) - higher is better
        """
        # Handle game end states
        if simulation.game_over:
            if simulation.is_win_condition():
                return (1.0,)  # Victory
            
            # More nuanced loss evaluation based on progress
            cures_discovered = len(getattr(simulation, 'discovered_cures', []))
            remaining_outbreaks = simulation.outbreak_limit - simulation.outbreak_count
            
            # More progress = better score even in loss
            progress_score = 0.1 + (0.1 * cures_discovered / 4.0) + (0.05 * remaining_outbreaks / simulation.outbreak_limit)
            return (max(0.1, min(0.3, progress_score)),)
        
        score = 0.0
        
        # 1. Cure Discovery Progress (35%)
        cures_discovered = len(getattr(simulation, 'discovered_cures', []))
        
        # Calculate potential cures based on all players' cards
        potential_cures = self._evaluate_potential_cures(simulation)
        
        cure_progress = cures_discovered / 4.0
        potential_progress = min(1.0, (cures_discovered + potential_cures) / 4.0)
        
        score += 0.25 * cure_progress  # Actual progress
        score += 0.10 * potential_progress  # Potential progress
        
        # 2. Infection Control (25%)
        infection_control_score = self._evaluate_infection_control(simulation)
        score += 0.25 * infection_control_score
        
        # 3. Outbreak Safety (20%)
        outbreak_count = simulation.outbreak_count
        outbreak_limit = simulation.outbreak_limit
        outbreak_safety = 1.0 - (outbreak_count / outbreak_limit)
        
        # Add penalty for cities at risk of outbreaks
        high_risk_cities = sum(1 for city in simulation.cities if getattr(city, 'infection_level', 0) >= 3)
        outbreak_risk_factor = max(0, 1.0 - (high_risk_cities * 0.15))
        
        score += 0.20 * (outbreak_safety * 0.7 + outbreak_risk_factor * 0.3)
        
        # 4. Research Station Strategic Placement (10%)
        research_station_score = self._evaluate_research_stations(simulation)
        score += 0.10 * research_station_score
        
        # 5. Team Positioning (10%)
        team_positioning_score = self._evaluate_team_positioning(simulation)
        score += 0.10 * team_positioning_score
        
        return (max(0.1, min(1.0, score)),)
    
    def _evaluate_potential_cures(self, simulation):
        """Evaluate how many cures could potentially be discovered based on current cards"""
        # Count cards by color across all players
        color_counts = defaultdict(int)
        cards_by_player_by_color = defaultdict(lambda: defaultdict(list))
        
        for player in simulation.players:
            for card in player.hand:
                if hasattr(card, 'color') and card.color and card.color != "INF":
                    color_counts[card.color] += 1
                    cards_by_player_by_color[player.id][card.color].append(card)
        
        # Get already discovered cures
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        
        # Count potential new cures (considering scientist role needs only 4 cards)
        potential_cures = 0
        for color, count in color_counts.items():
            if color in discovered_cures:
                continue
                
            # Check if any player is a scientist (needs only 4 cards)
            scientist_present = any(
                getattr(player, 'role', None) and getattr(player.role, 'name', '') == "Scientist"
                for player in simulation.players
            )
            
            cards_needed = 4 if scientist_present else 5
            
            # Basic check - do we have enough cards total?
            if count >= cards_needed:
                # Optimistic estimate: assume perfect card distribution is possible
                potential_cures += 0.8
            elif count >= cards_needed - 1:
                # Almost there
                potential_cures += 0.4
            elif count >= cards_needed - 2:
                # Making progress
                potential_cures += 0.2
        
        return potential_cures
    
    def _evaluate_infection_control(self, simulation):
        """Evaluate how well infections are being controlled"""
        # Calculate total infection level
        total_infection = sum(getattr(city, 'infection_level', 0) for city in simulation.cities)
        max_possible_infection = 3 * len(simulation.cities)  # Theoretical maximum
        
        # Basic infection control score
        base_control_score = 1.0 - (total_infection / max_possible_infection)
        
        # Identify infection clusters and hotspots
        hotspots = []
        infection_clusters = []
        
        for city in simulation.cities:
            if getattr(city, 'infection_level', 0) >= 2:
                hotspots.append(city)
                
                # Check if neighbors are also infected (cluster)
                infected_neighbors = sum(1 for neighbor in getattr(city, 'neighbours', [])
                                      if getattr(neighbor, 'infection_level', 0) > 0)
                if infected_neighbors >= 2:
                    infection_clusters.append((city, infected_neighbors))
        
        # Penalize for infection clusters (more dangerous than isolated infections)
        cluster_penalty = min(0.5, len(infection_clusters) * 0.1)
        
        # Save hotspots for decision making
        self.infection_hotspots = sorted(hotspots, 
                                        key=lambda city: getattr(city, 'infection_level', 0),
                                        reverse=True)
        
        return max(0.1, base_control_score - cluster_penalty)
    
    def _evaluate_research_stations(self, simulation):
        """Evaluate strategic placement of research stations"""
        research_stations = [city for city in simulation.cities if getattr(city, 'has_research_station', False)]
        
        # Base score on research station count
        count_score = min(1.0, len(research_stations) / 6.0)  # 6 research stations is optimal
        
        # No research stations is very bad
        if not research_stations:
            return 0.1
        
        # Evaluate geographic distribution
        # (This is a simplified proxy - ideally we'd use network connectivity measures)
        station_colors = Counter(getattr(city, 'color', 'unknown') for city in research_stations)
        distinct_colors = len(station_colors)
        distribution_score = distinct_colors / 4.0  # Ideally stations in all 4 regions
        
        # Return combined score
        return 0.4 * count_score + 0.6 * distribution_score
    
    def _evaluate_team_positioning(self, simulation):
        """Evaluate how well the team is positioned"""
        # Count players in high infection areas
        players_addressing_hotspots = 0
        for player in simulation.players:
            if player.city in self.infection_hotspots:
                players_addressing_hotspots += 1
        
        # Check for players near research stations
        players_near_stations = 0
        research_stations = [city for city in simulation.cities if getattr(city, 'has_research_station', False)]
        
        for player in simulation.players:
            if player.city in research_stations:
                players_near_stations += 1
            else:
                # Check if adjacent to research station
                for neighbor in getattr(player.city, 'neighbours', []):
                    if neighbor in research_stations:
                        players_near_stations += 0.5
                        break
        
        # Balance between addressing infections and being near research stations
        if self.infection_hotspots:
            hotspot_score = min(1.0, players_addressing_hotspots / min(len(self.infection_hotspots), len(simulation.players)))
        else:
            hotspot_score = 1.0  # No hotspots is good
            
        station_score = min(1.0, players_near_stations / len(simulation.players))
        
        # If we have cures discovered, weight toward infection control
        cures_discovered = len(getattr(simulation, 'discovered_cures', []))
        if cures_discovered >= 2:
            return 0.7 * hotspot_score + 0.3 * station_score
        else:
            # Otherwise, balance between both objectives
            return 0.5 * hotspot_score + 0.5 * station_score
    
    def _score_action(self, action, individual, player, simulation):
        """
        Enhanced action scoring with cooperative strategy awareness
        
        Returns:
            float: Action score (higher is better)
        """
        # Generate a cache key for this action evaluation
        action_type = action.get("type")
        player_id = getattr(player, 'id', id(player))
        
        # Basic values for all action types
        base_value = 0
        
        # ===== TREAT DISEASE =====
        if action_type == "treat":
            city = action.get("city", player.city)
            infection_level = getattr(city, 'infection_level', 0)
            
            if infection_level <= 0:
                return -1000  # Invalid or ineffective treatment
                
            # Critical infection gets highest priority
            if infection_level >= 3:
                base_value = 3000  # Extremely high value to prevent outbreaks
            elif infection_level == 2:
                base_value = 1500  # High priority
            else:
                base_value = 800  # Medium priority
                
            # Higher value if this city is a potential outbreak cascade risk
            risk_multiplier = 1.0
            infected_neighbors = sum(1 for neighbor in getattr(city, 'neighbours', [])
                                    if getattr(neighbor, 'infection_level', 0) >= 2)
            if infected_neighbors >= 2:
                risk_multiplier = 1.5  # Much higher priority if could trigger chain reaction
            
            base_value *= risk_multiplier
            
            # Adjust based on cure status - if cure exists, treating is less valuable
            discovered_cures = getattr(simulation, 'discovered_cures', [])
            city_color = getattr(city, 'color', None)
            
            if city_color in discovered_cures:
                # If cure exists and player is medic, maintain high priority (auto-treats)
                if getattr(player, 'role', None) and getattr(player.role, 'name', '') == "Medic":
                    pass  # Keep high priority
                else:
                    base_value *= 0.7  # Reduce priority if cure exists
        
        # ===== DISCOVER CURE =====
        elif action_type == "discover_cure":
            color = action.get("color")
            discovered = len(getattr(simulation, 'discovered_cures', []))
            
            # Discovering cure is highest priority objective
            base_value = 5000 - (discovered * 300)  # Still high priority but decreasing
            
            # Check if this color has high infection rate
            color_infection = sum(getattr(city, 'infection_level', 0) 
                                for city in simulation.cities 
                                if getattr(city, 'color', None) == color)
            
            # Higher value for curing colors with more infection
            if color_infection > 5:
                base_value += 500
        
        # ===== BUILD RESEARCH STATION =====
        elif action_type == "build":
            if player.city.has_research_station:
                base_value = -1000  # Invalid action
            else:
                # Count existing research stations
                research_stations = [city for city in simulation.cities if getattr(city, 'has_research_station', False)]
                
                # More strategic research station evaluation
                existing_station_count = len(research_stations)
                
                if existing_station_count == 0:
                    # First station is critical
                    base_value = 2500
                elif existing_station_count < 3:
                    # Early stations are important
                    base_value = 1800
                elif existing_station_count < 5:
                    # Mid-game stations are useful
                    base_value = 1200
                else:
                    # Late-game stations have diminishing returns
                    base_value = 600
                
                # Check if this location is strategic
                strategic_value = 0
                
                # Connects to many cities?
                neighbor_count = len(getattr(player.city, 'neighbours', []))
                strategic_value += min(300, neighbor_count * 50)
                
                # High infection area?
                area_infection = sum(getattr(city, 'infection_level', 0) 
                                   for city in getattr(player.city, 'neighbours', []))
                strategic_value += min(300, area_infection * 100)
                
                # Good distribution (different color from existing stations)?
                station_colors = [getattr(city, 'color', None) for city in research_stations]
                if getattr(player.city, 'color', None) not in station_colors:
                    strategic_value += 300
                
                base_value += strategic_value
        
        # ===== MOVE =====
        elif action_type == "move":
            target_city = action.get("target_city")
            
            if not target_city or target_city == player.city:
                return -2000  # Invalid move
            
            # Basic move has slight cost
            base_value = -50
            
            # Avoid revisiting recent cities
            if player_id in self.movement_history:
                recent_visits = self.movement_history[player_id]
                if target_city.name in recent_visits:
                    # Penalty grows with number of recent visits
                    visits = recent_visits.count(target_city.name)
                    recency_penalty = sum(500 for i, city in enumerate(reversed(recent_visits))
                                        if city == target_city.name and i < 3)
                    base_value -= (visits * 800 + recency_penalty)
            
            # Strategic move evaluation
            move_value = 0
            
            # Moving to treat infection
            if getattr(target_city, 'infection_level', 0) > 0:
                move_value += 300 + (getattr(target_city, 'infection_level', 0) * 200)
                
                # Extra value if target is at outbreak risk
                if getattr(target_city, 'infection_level', 0) >= 3:
                    move_value += 1000
            
            # Moving to or near research station (for cure discovery)
            if getattr(target_city, 'has_research_station', False):
                move_value += 200
                
                # Extra value if player has cards for cure
                color_counts = Counter(getattr(card, 'color', None) for card in player.hand
                                     if hasattr(card, 'color') and card.color and card.color != "INF")
                discovered_cures = getattr(simulation, 'discovered_cures', [])
                
                for color, count in color_counts.items():
                    if color not in discovered_cures:
                        cards_needed = 4 if (getattr(player, 'role', None) and 
                                           getattr(player.role, 'name', '') == "Scientist") else 5
                        if count >= cards_needed - 1:  # Close to having a cure
                            move_value += 800
                        elif count >= cards_needed - 2:  # Making progress
                            move_value += 300
            
            # Moving to facilitate card exchange
            for other_player in simulation.players:
                if other_player.id != player.id and other_player.city == target_city:
                    # Cooperative strategy: consider card exchange possibilities
                    exchange_value = self._evaluate_card_exchange(player, other_player, simulation)
                    move_value += exchange_value
            
            # Moving to high-risk areas
            target_is_hotspot = target_city in self.infection_hotspots
            if target_is_hotspot:
                move_value += 400
            
            # Moving to build research station
            can_build = any(getattr(card, 'city_name', None) == target_city.name for card in player.hand)
            if can_build and not getattr(target_city, 'has_research_station', False):
                research_stations = [city for city in simulation.cities if getattr(city, 'has_research_station', False)]
                if len(research_stations) < 6:  # Max stations
                    move_value += 500
            
            base_value += move_value
        
        # ===== SHARE KNOWLEDGE =====
        elif action_type == "share_knowledge":
            # Highly valuable cooperative action
            card = action.get("card")
            recipient = action.get("recipient")
            
            if not card or not recipient:
                return -1000
            
            # Base value for card sharing
            base_value = 500
            
            # Strategic card sharing evaluation
            if hasattr(card, 'color') and card.color and card.color != "INF":
                # Check if this helps complete a set for cure discovery
                recipient_cards = Counter(getattr(c, 'color', None) for c in recipient.hand
                                     if hasattr(c, 'color') and c.color and c.color != "INF")
                discovered_cures = getattr(simulation, 'discovered_cures', [])
                
                color = card.color
                if color not in discovered_cures:
                    current_count = recipient_cards.get(color, 0)
                    cards_needed = 4 if (getattr(recipient, 'role', None) and 
                                      getattr(recipient.role, 'name', '') == "Scientist") else 5
                    
                    # Very high value if this completes a set
                    if current_count == cards_needed - 1:
                        base_value += 2000
                    # High value if this gets closer
                    elif current_count >= cards_needed - 3:
                        base_value += 1000
                    else:
                        base_value += 200
        
        # ===== PASS =====
        elif action_type == "pass":
            # Passing is generally bad
            base_value = -2000
        
        # ===== DISCARD =====
        elif action_type == "discard":
            # Baseline for discarding
            base_value = -500
            
            card = action.get("card")
            if not card:
                return -500
                
            # Strategic discard evaluation
            if hasattr(card, 'color') and card.color and card.color != "INF":
                color = card.color
                discovered_cures = getattr(simulation, 'discovered_cures', [])
                
                # Discard cards of colors already cured
                if color in discovered_cures:
                    base_value += 200
                else:
                    # Count how many of this color we have
                    color_count = sum(1 for c in player.hand 
                                    if hasattr(c, 'color') and getattr(c, 'color', None) == color)
                    
                    # Penalize discarding rare cards or cards needed for cures
                    cards_needed = 4 if (getattr(player, 'role', None) and 
                                      getattr(player.role, 'name', '') == "Scientist") else 5
                    
                    if color_count <= 2:
                        base_value -= 100  # Penalty for discarding rare colors
                    elif color_count >= cards_needed:
                        base_value += 100  # OK to discard extras
                    
                    # Check if others have this color (team perspective)
                    team_has_color = False
                    for other_player in simulation.players:
                        if other_player.id != player.id:
                            for c in other_player.hand:
                                if hasattr(c, 'color') and getattr(c, 'color', None) == color:
                                    team_has_color = True
                                    break
                    
                    if not team_has_color:
                        base_value -= 200  # Big penalty if team has none of this color
        
        # ===== Role-specific bonuses =====
        role_bonus = 0
        if hasattr(player, 'role') and player.role:
            role_name = getattr(player.role, 'name', '')
            
            if role_name == "Medic" and action_type == "treat":
                # Medic treats are extremely efficient
                role_bonus += 800
            
            elif role_name == "Scientist" and action_type == "discover_cure":
                # Scientist has advantage for cure discovery
                role_bonus += 1000
            
            elif role_name == "Operations_Expert" and action_type == "build":
                # Operations Expert builds research stations easily
                role_bonus += 700
                if not action.get("card"):
                    role_bonus += 300
                
            elif role_name == "Researcher" and action_type == "share_knowledge":
                # Researcher excels at sharing knowledge
                role_bonus += 600
                card = action.get("card")
                if card and hasattr(card, 'city_name}') and card.city_name != player.city.name:
                    role_bonus += 300
        
        # Return final score
        return base_value + role_bonus
    
    def _evaluate_card_exchange(self, player1, player2, simulation):
        """Evaluate the strategic value of card exchange between players"""
        # If players aren't in same city, exchange isn't possible
        if player1.city != player2.city:
            return 0
            
        exchange_value = 0
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        
        # Get color counts for both players
        p1_colors = Counter(getattr(card, 'color', None) for card in player1.hand
                          if hasattr(card, 'color') and card.color and card.color != "INF")
        
        p2_colors = Counter(getattr(card, 'color', None) for card in player2.hand
                          if hasattr(card, 'color') and card.color and card.color != "INF")
        
        # Check possible card combinations that could lead to cures
        for color in set(list(p1_colors.keys()) + list(p2_colors.keys())):
            if color in discovered_cures:
                continue
                
            # Check if exchanging would help complete a set
            p1_count = p1_colors.get(color, 0)
            p2_count = p2_colors.get(color, 0)
            
            # Is player 2 a scientist?
            p2_is_scientist = (getattr(player2, 'role', None) and 
                            getattr(player2.role, 'name', '') == "Scientist")
            
            p2_cards_needed = 4 if p2_is_scientist else 5
            
            # Evaluate benefit of player1 giving cards to player2
            if p1_count > 0 and p2_count >= p2_cards_needed - 2:
                # High value if close to complete set
                exchange_value += 500 + (p2_count * 100)
            
            # Is player 1 a scientist?
            p1_is_scientist = (getattr(player1, 'role', None) and 
                            getattr(player1.role, 'name', '') == "Scientist")
            
            p1_cards_needed = 4 if p1_is_scientist else 5
            
            # Evaluate benefit of player2 giving cards to player1
            if p2_count > 0 and p1_count >= p1_cards_needed - 2:
                # High value if close to complete set
                exchange_value += 500 + (p1_count * 100)
        
        return exchange_value
    
    def decide_action(self, player, simulation):
        """
        Enhanced decision making with strategic and cooperative planning
        
        Args:
            player: The player making the decision
            simulation: The current game state
            
        Returns:
            dict: The chosen action
        """
        # 1. Emergency response - handle critical situations first
        emergency_action = self._handle_emergencies(player, simulation)
        if emergency_action:
            return emergency_action
        
        # 2. Update game state assessment
        self._update_game_assessment(player, simulation)
        
        # 3. Hand limit management
        if len(player.hand) > 7:
            return {"type": "discard", "card": self._find_best_discard(player, simulation)}
        
        # 4. Strategic objective selection based on game state
        objective = self._select_strategic_objective(player, simulation)
        
        # 5. Cure discovery opportunity check
        cure_action = self._try_discover_cure(player, simulation)
        if cure_action:
            return cure_action
        
        # 6. Cooperative action opportunity check
        coop_action = self._plan_cooperative_action(player, simulation)
        if coop_action:
            return coop_action
        
        # 7. Generate and evaluate action chains
        action_chains = self._generate_action_chains(player, simulation)
        
        if not action_chains:
            return {"type": "pass"}
            
        # 8. Choose best action chain
        best_chain = None
        best_score = float('-inf')
        
        for chain in action_chains:
            if self._is_chain_aligned_with_objective(chain, objective):
                # Bonus for chains aligned with strategic objective
                chain_bonus = 500
            else:
                chain_bonus = 0
                
            score = self._evaluate_action_chain(chain, player, simulation) + chain_bonus
            if score > best_score:
                best_score = score
                best_chain = chain
        
        # 9. Execute first action in best chain
        if best_chain and best_chain[0]:
            first_action = best_chain[0]
            
            # Update movement history if applicable
            if first_action.get("type") == "move":
                self._update_movement_history(player, first_action)
            
            return first_action
        
        # Default: pass if no good actions found
        return {"type": "pass"}
    
    def _handle_emergencies(self, player, simulation):
        """Handle emergency situations requiring immediate response"""
        # Threat level assessment - critical infection first
        if player.city.infection_level >= 3:
            return {"type": "treat", "city": player.city}
        
        # Check for imminent outbreaks in adjacent cities
        for neighbor in player.city.neighbours:
            if getattr(neighbor, 'infection_level', 0) >= 3:
                return {"type": "move", "target_city": neighbor}
        
        # Check if we've exceeded hand limit
        if len(player.hand) > 7:
            return {"type": "discard", "card": self._find_best_discard(player, simulation)}
        
        # Complete cure if possible and already at research station
        if player.city.has_research_station:
            for color, cards in self._group_cards_by_color(player.hand).items():
                discovered_cures = getattr(simulation, 'discovered_cures', [])
                if color not in discovered_cures:
                    cards_needed = 4 if (getattr(player, 'role', None) and 
                                      getattr(player.role, 'name', '') == "Scientist") else 5
                    if len(cards) >= cards_needed:
                        return {
                            "type": "discover_cure",
                            "color": color,
                            "cards": cards[:cards_needed]
                        }
        
        return None
    
    def _update_game_assessment(self, player, simulation):
        """Update global game state assessment"""
        # Update infection risk assessment
        self._evaluate_infection_control(simulation)
        
        # Track cure progress
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        self.cure_progress = {color: 0 for color in ['Blue', 'Yellow', 'Black', 'Red']}
        
        # Mark discovered cures
        for color in discovered_cures:
            self.cure_progress[color] = 1.0
            
        # Assess progress toward undiscovered cures
        for color in self.cure_progress:
            if self.cure_progress[color] == 1.0:
                continue
                
            # Count cards of this color across all players
            total_cards = sum(1 for player in simulation.players
                           for card in player.hand
                           if hasattr(card, 'color') and getattr(card, 'color', None) == color)
            
            # Approximation of progress (5 cards needed, scientist needs 4)
            scientist_present = any(getattr(player, 'role', None) and 
                                   getattr(player.role, 'name', '') == "Scientist"
                                   for player in simulation.players)
            
            cards_needed = 4 if scientist_present else 5
            self.cure_progress[color] = min(0.99, total_cards / cards_needed)
        
        # Update team strategy based on game state
        self._update_team_strategy(player, simulation)
    
    def _update_team_strategy(self, player, simulation):
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
            # Early game: establish research infrastructure and control initial spread
            self.team_strategy = {
                "priority": "research_infrastructure" if outbreak_risk < 0.3 else "control_spread",
                "secondary": "cure_preparation"
            }
        else:
            # Mid game: balance cure progress and infection control
            self.team_strategy = {
                "priority": "cure_progress" if outbreak_risk < 0.5 else "control_critical",
                "secondary": "optimize_positioning"
            }
    
    def _select_strategic_objective(self, player, simulation):
        """Select a strategic objective based on game state and team strategy"""
        # Get current strategy
        priority = self.team_strategy.get("priority", "cure_progress")
        
        # Role-specific adjustments
        role_name = getattr(player.role, 'name', '') if hasattr(player, 'role') else ''
        
        # Customize objective based on role and priority
        if role_name == "Medic":
            if priority in ["control_outbreaks", "control_critical", "control_spread"]:
                return "treat_infections"
            else:
                return "position_for_cures"
                
        elif role_name == "Scientist":
            return "discover_cures"
            
        elif role_name == "Operations_Expert":
            if priority == "research_infrastructure":
                return "build_stations"
            else:
                return "support_team"
                
        elif role_name == "Researcher":
            return "share_knowledge"
        
        # Default objectives based on team priority
        if priority == "final_cure":
            return "discover_cures"
        elif priority == "control_outbreaks":
            return "treat_infections"
        elif priority == "research_infrastructure":
            return "build_stations"
        elif priority == "cure_progress":
            return "collect_cards"
        else:
            return "balanced_approach"
    
    def _is_chain_aligned_with_objective(self, chain, objective):
        """Check if an action chain aligns with the strategic objective"""
        if not chain:
            return False
            
        first_action = chain[0]
        action_type = first_action.get("type")
        
        # Check alignment based on objective
        if objective == "treat_infections" and action_type in ["treat", "move"]:
            # Move actions may be part of treatment strategy
            if action_type == "move":
                target_city = first_action.get("target_city")
                if target_city and getattr(target_city, 'infection_level', 0) > 0:
                    return True
            else:
                return True
                
        elif objective == "discover_cures" and action_type in ["discover_cure", "move", "build"]:
            # Moving to research station or building one aligns with cure objective
            if action_type == "move":
                target_city = first_action.get("target_city")
                if target_city and getattr(target_city, 'has_research_station', False):
                    return True
            else:
                return True
                
        elif objective == "build_stations" and action_type in ["build", "move"]:
            # Moving to build stations aligns with infrastructure objective
            if action_type == "move":
                target_city = first_action.get("target_city")
                if target_city and not getattr(target_city, 'has_research_station', False):
                    return True
            else:
                return True
                
        elif objective == "collect_cards" and action_type in ["move", "share_knowledge"]:
            return True
            
        elif objective == "share_knowledge" and action_type in ["move", "share_knowledge"]:
            return True
            
        elif objective == "balanced_approach":
            return True
        
        return False
    
    def _find_best_discard(self, player, simulation):
        """Find the card with least strategic value to discard"""
        if not player.hand:
            return None
            
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        
        # First priority: discard event cards if no urgent use
        event_cards = [card for card in player.hand if getattr(card, 'type', '') == 'event']
        if event_cards:
            return event_cards[0]
        
        # Second priority: discard cards of colors already cured
        cured_color_cards = [card for card in player.hand 
                            if hasattr(card, 'color') and card.color in discovered_cures]
        if cured_color_cards:
            return cured_color_cards[0]
        
        # Group cards by color
        cards_by_color = self._group_cards_by_color(player.hand)
        
        # Find color with most cards (can afford to lose one)
        most_common_color = None
        most_common_count = 0
        
        for color, cards in cards_by_color.items():
            if len(cards) > most_common_count and color not in discovered_cures:
                most_common_color = color
                most_common_count = len(cards)
        
        # If we have surplus of a color, discard one
        if most_common_color and most_common_count > 5:
            return cards_by_color[most_common_color][0]
        
        # Last resort: discard city card that doesn't match player's current city
        for card in player.hand:
            if (hasattr(card, 'city_name') and card.city_name != player.city.name):
                return card
        
        # If all else fails, discard first card
        return player.hand[0]
    
    def _try_discover_cure(self, player, simulation):
        """Attempt to discover a cure if conditions are favorable"""
        # Must be at research station
        if not player.city.has_research_station:
            return None
            
        # Group cards by color
        cards_by_color = self._group_cards_by_color(player.hand)
        
        # Check discovered cures
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        
        # Determine cards needed (4 for Scientist, 5 for others)
        cards_needed = 4 if (getattr(player, 'role', None) and 
                           getattr(player.role, 'name', '') == "Scientist") else 5
        
        # Find viable colors for cure discovery
        viable_cures = []
        for color, cards in cards_by_color.items():
            if color not in discovered_cures and len(cards) >= cards_needed:
                # Calculate strategic priority based on infection levels
                color_infection = sum(getattr(city, 'infection_level', 0) 
                                   for city in simulation.cities 
                                   if getattr(city, 'color', None) == color)
                
                viable_cures.append((color, cards, color_infection))
        
        # Sort by infection level (prioritize more infected colors)
        viable_cures.sort(key=lambda x: x[2], reverse=True)
        
        # If any viable cures, discover the highest priority one
        if viable_cures:
            color, cards, _ = viable_cures[0]
            return {
                "type": "discover_cure",
                "color": color,
                "cards": cards[:cards_needed]
            }
        
        return None
    
    def _plan_cooperative_action(self, player, simulation):
        """Plan actions that facilitate team cooperation"""
        # Check for knowledge sharing opportunities
        for other_player in simulation.players:
            if other_player.id != player.id and other_player.city == player.city:
                # Check if we can share valuable cards
                share_action = self._evaluate_knowledge_sharing(player, other_player, simulation)
                if share_action:
                    return share_action
        
        # Check if we should move to facilitate knowledge sharing
        if len(player.hand) > 0:
            for other_player in simulation.players:
                if other_player.id != player.id and other_player.city != player.city:
                    # Evaluate benefit of moving to other player's location
                    move_for_sharing = self._evaluate_move_for_sharing(player, other_player, simulation)
                    if move_for_sharing:
                        return move_for_sharing
        
        return None
    
    def _evaluate_knowledge_sharing(self, player, other_player, simulation):
        """Evaluate opportunities for strategic knowledge sharing"""
        # Both players must be in same city
        if player.city != other_player.city:
            return None
            
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        
        # Check if researcher (can share any card)
        player_is_researcher = (getattr(player, 'role', None) and 
                              getattr(player.role, 'name', '') == "Researcher")
        
        other_is_researcher = (getattr(other_player, 'role', None) and 
                             getattr(other_player.role, 'name', '') == "Researcher")
        
        # Get color counts for both players
        player_colors = Counter()
        for card in player.hand:
            if hasattr(card, 'color') and card.color and card.color != "INF":
                player_colors[card.color] += 1
        
        other_colors = Counter()
        for card in other_player.hand:
            if hasattr(card, 'color') and card.color and card.color != "INF":
                other_colors[card.color] += 1
        
        # Check if sharing would help complete a set
        best_share = None
        best_value = 0
        
        # Can player share with other?
        for card in player.hand:
            if hasattr(card, 'city_name') and card.city_name == player.city.name or player_is_researcher:
                if hasattr(card, 'color') and card.color and card.color != "INF":
                    color = card.color
                    
                    if color not in discovered_cures:
                        # How close is other player to completing set?
                        other_count = other_colors.get(color, 0)
                        other_needed = 4 if (getattr(other_player, 'role', None) and 
                                          getattr(other_player.role, 'name', '') == "Scientist") else 5
                        
                        if other_count + 1 >= other_needed:
                            # This completes a set!
                            share_value = 2000
                        elif other_count + 1 >= other_needed - 1:
                            # Gets very close
                            share_value = 1000
                        elif other_count + 1 >= other_needed - 2:
                            # Makes good progress
                            share_value = 500
                        else:
                            # Minimal progress
                            share_value = 200
                        
                        if share_value > best_value:
                            best_value = share_value
                            best_share = {
                                "type": "share_knowledge",
                                "card": card,
                                "recipient": other_player
                            }
        
        # Can other player share with player?
        for card in other_player.hand:
            if hasattr(card, 'city_name') and card.city_name == player.city.name or other_is_researcher:
                if hasattr(card, 'color') and card.color and card.color != "INF":
                    color = card.color
                    
                    if color not in discovered_cures:
                        # How close is player to completing set?
                        player_count = player_colors.get(color, 0)
                        player_needed = 4 if (getattr(player, 'role', None) and 
                                           getattr(player.role, 'name', '') == "Scientist") else 5
                        
                        if player_count + 1 >= player_needed:
                            # This completes a set!
                            share_value = 2000
                        elif player_count + 1 >= player_needed - 1:
                            # Gets very close
                            share_value = 1000
                        elif player_count + 1 >= player_needed - 2:
                            # Makes good progress
                            share_value = 500
                        else:
                            # Minimal progress
                            share_value = 200
                        
                        if share_value > best_value:
                            best_value = share_value
                            best_share = {
                                "type": "share_knowledge",
                                "card": card,
                                "giver": other_player,
                                "receiver": player
                            }
        
        # Return best sharing action if valuable enough
        if best_value >= 500:  # Only share if significant value
            return best_share
            
        return None
    
    def _evaluate_move_for_sharing(self, player, other_player, simulation):
        """Evaluate if moving to another player would enable valuable card sharing"""
        # Only consider if close enough to reach
        if player.city not in other_player.city.neighbours:
            return None
            
        discovered_cures = getattr(simulation, 'discovered_cures', [])
        
        # Get player's cards by color
        player_colors = Counter()
        for card in player.hand:
            if hasattr(card, 'color') and card.color and card.color != "INF":
                player_colors[card.color] += 1
        
        # Get other player's cards by color
        other_colors = Counter()
        for card in other_player.hand:
            if hasattr(card, 'color') and card.color and card.color != "INF":
                other_colors[card.color] += 1
        
        # Check if moving would enable valuable sharing
        move_value = 0
        
        # Check colors not yet cured
        for color in set(list(player_colors.keys()) + list(other_colors.keys())):
            if color in discovered_cures:
                continue
                
            player_count = player_colors.get(color, 0)
            other_count = other_colors.get(color, 0)
            
            # Check if either player is scientist
            player_is_scientist = (getattr(player, 'role', None) and 
                                 getattr(player.role, 'name', '') == "Scientist")
            other_is_scientist = (getattr(other_player, 'role', None) and 
                                getattr(other_player.role, 'name', '') == "Scientist")
            
            player_needed = 4 if player_is_scientist else 5
            other_needed = 4 if other_is_scientist else 5
            
            # Could player complete other's set?
            if player_count > 0 and other_count >= other_needed - 2:
                move_value += 500 + (other_count * 100)
            
            # Could other complete player's set?
            if other_count > 0 and player_count >= player_needed - 2:
                move_value += 500 + (player_count * 100)
        
        # If valuable enough, move to other player
        if move_value >= 800:
            return {
                "type": "move",
                "target_city": other_player.city
            }
            
        return None
    
    def _generate_action_chains(self, player, simulation, depth=2):
        """Generate chains of actions with optimized search"""
        chains = []
        
        # Get possible actions
        possible_actions = self.get_possible_actions(player, simulation)
        
        # Filter valid actions
        valid_actions = [action for action in possible_actions 
                        if self._is_valid_action(action, player, simulation)]
        
        # Add single actions to chains
        for action in valid_actions:
            chains.append([action])
        
        # Generate deeper chains if needed
        if depth >= 2 and getattr(player, 'remaining_actions', 4) > 1:
            # Score first actions to prioritize exploration
            scored_actions = []
            for action in valid_actions:
                score = self._score_action(action, self.best_individual, player, simulation)
                if score > -500:  # Filter out very low value actions
                    scored_actions.append((action, score))
            
            # Sort by score and take top actions
            scored_actions.sort(key=lambda x: x[1], reverse=True)
            top_k = min(5, max(3, len(scored_actions)))
            
            # Explore chains from top actions
            for action, _ in scored_actions[:top_k]:
                # Simulate this action
                sim_copy = self._simulate_action_effect(action, player, simulation)
                if not sim_copy:
                    continue
                
                # Get next possible actions
                remaining_actions = getattr(player, 'remaining_actions', 4) - 1
                if remaining_actions <= 0:
                    continue
                
                next_possible_actions = self.get_possible_actions(player, sim_copy)
                next_valid_actions = [a for a in next_possible_actions
                                    if self._is_valid_action(a, player, sim_copy)]
                
                # Score and filter next actions
                next_scored_actions = []
                for next_action in next_valid_actions:
                    score = self._score_action(next_action, self.best_individual, player, sim_copy)
                    if score > -300:  # Stricter filter for second actions
                        next_scored_actions.append((next_action, score))
                
                next_scored_actions.sort(key=lambda x: x[1], reverse=True)
                next_top_k = min(3, len(next_scored_actions))
                
                # Add chains with second actions
                for next_action, _ in next_scored_actions[:next_top_k]:
                    chains.append([action, next_action])
        
        return chains
    
    def _evaluate_action_chain(self, chain, player, simulation):
        """Evaluate action chain with strategic awareness"""
        if not chain:
            return 0
        
        # Create cache key
        cache_key = str([(a.get("type"), getattr(a.get("target_city"), "name", None) 
                        if a.get("type") == "move" else None) for a in chain])
        
        # Check cache
        if cache_key in self._chain_eval_cache:
            return self._chain_eval_cache[cache_key]
        
        # Evaluate chain
        total_score = 0
        current_sim = simulation
        discount = 1.0
        
        for i, action in enumerate(chain):
            # Score individual action
            action_score = self._score_action(action, self.best_individual, player, current_sim)
            
            # Calculate synergy bonus
            chain_bonus = self._calculate_chain_synergy(chain[:i+1], player, current_sim)
            
            # Apply discounted score
            total_score += discount * (action_score + chain_bonus)
            
            # Simulate action effect
            next_sim = self._simulate_action_effect(action, player, current_sim)
            if not next_sim:
                break
                
            # Update for next iteration
            current_sim = next_sim
            discount *= 0.8  # Future actions worth less
        
        # Cache result
        if len(self._chain_eval_cache) > 1000:  # Limit cache size
            self._chain_eval_cache = {}
        self._chain_eval_cache[cache_key] = total_score
        
        return total_score
    
    def _calculate_chain_synergy(self, action_sequence, player, simulation):
        """Calculate synergy bonus for action sequences"""
        if len(action_sequence) < 2:
            return 0
            
        bonus = 0
        
        # Pattern: Move -> Treat (High infection)
        if (action_sequence[-2].get("type") == "move" and 
            action_sequence[-1].get("type") == "treat"):
            target_city = action_sequence[-2].get("target_city")
            treat_city = action_sequence[-1].get("city", player.city)
            
            if target_city and target_city.name == treat_city.name:
                infection_level = getattr(treat_city, "infection_level", 0)
                bonus += 300 + (infection_level * 200)
        
        # Pattern: Move -> Build
        elif (action_sequence[-2].get("type") == "move" and 
              action_sequence[-1].get("type") == "build"):
            # Check if strategic location
            target_city = action_sequence[-2].get("target_city")
            research_stations = [city for city in simulation.cities 
                               if getattr(city, "has_research_station", False)]
            
            # More valuable if no other stations nearby
            station_nearby = any(rs in getattr(target_city, "neighbours", []) 
                              for rs in research_stations)
            
            bonus += 500 if not station_nearby else 300
        
        # Pattern: Build -> Discover Cure
        elif (action_sequence[-2].get("type") == "build" and 
              action_sequence[-1].get("type") == "discover_cure"):
            bonus += 1000  # Very valuable sequence
        
        # Pattern: Move -> Research Station -> Discover Cure
        if len(action_sequence) >= 3:
            if (action_sequence[-3].get("type") == "move" and 
                action_sequence[-2].get("type") == "build" and
                action_sequence[-1].get("type") == "discover_cure"):
                bonus += 1500  # Extremely valuable
        
        return bonus
    
    def _simulate_action_effect(self, action, player, simulation):
        """Simulate the effect of an action on the game state"""
        # Create lightweight simulation state proxy
        sim_copy = type('SimProxy', (), {
            'original': simulation,
            'cities': simulation.cities,
            'players': simulation.players,
            'discovered_cures': list(getattr(simulation, 'discovered_cures', [])),
            'outbreak_count': simulation.outbreak_count,
            'game_over': simulation.game_over,
            'is_win_condition': simulation.is_win_condition
        })
        
        # Simulate action
        action_type = action.get("type")
        
        if action_type == "move":
            # Simulate move
            target_city = action.get("target_city")
            if not target_city:
                return None
                
            # Update player city
            setattr(player, '_simulated_city', target_city)
            
            # Override player.city property for simulation
            sim_copy.get_city = lambda p: getattr(p, '_simulated_city', p.city)
            
        elif action_type == "treat":
            # Simulate treatment
            city = action.get("city", player.city)
            infection_level = getattr(city, 'infection_level', 0)
            
            if infection_level <= 0:
                return None
                
            # Update infection level
            setattr(city, '_simulated_infection', max(0, infection_level - 1))
            
            # Override infection_level property
            sim_copy.get_infection = lambda c: getattr(c, '_simulated_infection', 
                                                     getattr(c, 'infection_level', 0))
        
        elif action_type == "build":
            # Simulate building research station
            if getattr(player.city, 'has_research_station', False):
                return None
                
            # Update research station
            setattr(player.city, '_simulated_station', True)
            
            # Override has_research_station property
            sim_copy.has_station = lambda c: getattr(c, '_simulated_station', False) or getattr(c, 'has_research_station', False)
            
        elif action_type == "discover_cure":
            # Simulate cure discovery
            color = action.get("color")
            if not color or color in sim_copy.discovered_cures:
                return None
                
            # Update discovered cures
            sim_copy.discovered_cures.append(color)
            
        elif action_type == "share_knowledge":
            # Knowledge sharing doesn't affect state evaluation
            pass
            
        return sim_copy
    
    def _group_cards_by_color(self, cards):
        """Group cards by color"""
        cards_by_color = {}
        for card in cards:
            if hasattr(card, 'color') and card.color and card.color != "INF":
                color = card.color
                if color not in cards_by_color:
                    cards_by_color[color] = []
                cards_by_color[color].append(card)
        return cards_by_color
    
    def _update_movement_history(self, player, action):
        """Update movement history for a player"""
        if action.get("type") == "move":
            target_city = action.get("target_city")
            if target_city:
                # Initialize history structures if needed
                if not hasattr(self, 'movement_history'):
                    self.movement_history = {}
                if not hasattr(self, 'previous_cities'):
                    self.previous_cities = {}
                
                # Get player ID
                player_id = getattr(player, 'id', id(player))
                
                # Update history
                if player_id not in self.movement_history:
                    self.movement_history[player_id] = []
                
                # Add to history and limit size
                self.movement_history[player_id].append(target_city.name)
                if len(self.movement_history[player_id]) > 5:
                    self.movement_history[player_id].pop(0)
                
                # Update previous city
                self.previous_cities[player_id] = target_city.name
    
    def _is_valid_action(self, action, player, simulation):
        """Check if an action is valid in the current state"""
        action_type = action.get("type")
        
        if action_type == "build":
            # Check if already has research station
            if getattr(player.city, 'has_research_station', False):
                return False
                
            # Check if has city card unless Operations Expert
            is_operations_expert = (getattr(player, 'role', None) and 
                                   getattr(player.role, 'name', '') == "Operations_Expert")
                                   
            if not is_operations_expert:
                has_city_card = any(
                    hasattr(card, 'city_name') and card.city_name == player.city.name
                    for card in player.hand
                )
                if not has_city_card:
                    return False
        
        elif action_type == "discover_cure":
            # Check if at research station
            if not getattr(player.city, 'has_research_station', False):
                return False
                
            # Check if has enough cards
            color = action.get("color")
            cards = action.get("cards", [])
            
            # Check if cure already discovered
            discovered_cures = getattr(simulation, 'discovered_cures', [])
            if color in discovered_cures:
                return False
                
            # Check card count
            required_cards = 4 if (getattr(player, 'role', None) and 
                                getattr(player.role, 'name', '') == "Scientist") else 5
            if len(cards) < required_cards:
                return False
        
        elif action_type == "treat":
            # Check if city has infection
            city = action.get("city", player.city)
            if getattr(city, 'infection_level', 0) <= 0:
                return False
                
        elif action_type == "move":
            # Check if target is valid
            target_city = action.get("target_city")
            if not target_city or target_city == player.city:
                return False
                
            # Check move method
            method = action.get("method", "standard")
            
            if method == "standard":
                # Check if neighbor
                if target_city not in getattr(player.city, 'neighbours', []):
                    return False
            elif method == "direct_flight":
                # Check if has city card
                has_card = any(
                    hasattr(card, 'city_name') and card.city_name == target_city.name
                    for card in player.hand
                )
                if not has_card:
                    return False
            elif method == "charter_flight":
                # Check if has current city card
                has_card = any(
                    hasattr(card, 'city_name') and card.city_name == player.city.name
                    for card in player.hand
                )
                if not has_card:
                    return False
            elif method == "shuttle_flight":
                # Check if both cities have research stations
                if not (getattr(player.city, 'has_research_station', False) and 
                       getattr(target_city, 'has_research_station', False)):
                    return False
        
        elif action_type == "share_knowledge":
            # Check if players in same city
            recipient = action.get("recipient")
            if not recipient or recipient.city != player.city:
                return False
                
            # Check if has city card
            card = action.get("card")
            if not card:
                return False
                
            # Check if card matches city or player is researcher
            is_researcher = (getattr(player, 'role', None) and 
                           getattr(player.role, 'name', '') == "Researcher")
            
            if not is_researcher and not (hasattr(card, 'city_name') and 
                                        card.city_name == player.city.name):
                return False
        
        return True
    
    def get_possible_actions(self, player, simulation):
        """Get all possible actions for a player"""
        actions = []
        
        # MOVE actions
        # Standard move to adjacent cities
        for city in getattr(player.city, 'neighbours', []):
            actions.append({
                "type": "move",
                "target_city": city,
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
        
        # Charter flight (using current city card)
        current_city_card = next((card for card in player.hand 
                               if hasattr(card, 'city_name') and card.city_name == player.city.name), None)
        if current_city_card:
            for city in simulation.cities:
                if city != player.city:
                    actions.append({
                        "type": "move",
                        "target_city": city,
                        "method": "charter_flight",
                        "card": current_city_card
                    })
        
        # Shuttle flight (between research stations)
        if getattr(player.city, 'has_research_station', False):
            research_stations = [city for city in simulation.cities 
                                if getattr(city, 'has_research_station', False) and city != player.city]
            for city in research_stations:
                actions.append({
                    "type": "move",
                    "target_city": city,
                    "method": "shuttle_flight"
                })
        
        # TREAT DISEASE action
        if getattr(player.city, 'infection_level', 0) > 0:
            actions.append({
                "type": "treat",
                "city": player.city
            })
        
        # BUILD RESEARCH STATION action
        if not getattr(player.city, 'has_research_station', False):
            # Check if Operations Expert (special ability)
            is_operations_expert = (getattr(player, 'role', None) and 
                                 getattr(player.role, 'name', '') == "Operations_Expert")
            
            city_card = next((card for card in player.hand 
                           if hasattr(card, 'city_name') and card.city_name == player.city.name), None)
            
            if city_card or is_operations_expert:
                actions.append({
                    "type": "build",
                    "city": player.city,
                    "card": city_card
                })
        
        # DISCOVER CURE action
        if getattr(player.city, 'has_research_station', False):
            # Group cards by color
            cards_by_color = self._group_cards_by_color(player.hand)
            
            # Check discovered cures
            discovered_cures = getattr(simulation, 'discovered_cures', [])
            
            # Determine cards needed (4 for Scientist, 5 for others)
            cards_needed = 4 if (getattr(player, 'role', None) and 
                               getattr(player.role, 'name', '') == "Scientist") else 5
            
            # Add cure actions for each viable color
            for color, cards in cards_by_color.items():
                if color not in discovered_cures and len(cards) >= cards_needed:
                    actions.append({
                        "type": "discover_cure",
                        "color": color,
                        "cards": cards[:cards_needed]
                    })
        
        # SHARE KNOWLEDGE action
        # Find players in same city
        same_city_players = [p for p in simulation.players if p.id != player.id and p.city == player.city]
        
        if same_city_players:
            # Check if player is researcher (can share any card)
            is_researcher = (getattr(player, 'role', None) and 
                           getattr(player.role, 'name', '') == "Researcher")
            
            # Cards player can share
            sharable_cards = []
            if is_researcher:
                sharable_cards = player.hand
            else:
                # Can only share cards matching current city
                sharable_cards = [card for card in player.hand 
                                if hasattr(card, 'city_name') and card.city_name == player.city.name]
            
            # Add share actions
            for card in sharable_cards:
                for other_player in same_city_players:
                    actions.append({
                        "type": "share_knowledge",
                        "card": card,
                        "recipient": other_player
                    })
        
        # PASS action
        actions.append({
            "type": "pass"
        })
        
        return actions

# Function to be used as strategy in the game
def ea_agent_strategy(player):
    """Strategy function using the  EA agent"""
    global _global_ea_agent
    
    # Initialize agent if needed
    import os
    agent_state_dir = "./agents_state"
    os.makedirs(agent_state_dir, exist_ok=True)
    state_file = os.path.join(agent_state_dir, "eaagent_state.pkl")
    
    if '_global_ea_agent' not in globals():
        global _global_ea_agent
        _global_ea_agent = EAAgent()
        print(f"Created new  EA agent (state file: {state_file})")
        # Try to load previous state
        try:
            with open(state_file, "rb") as f:
                saved_state = pickle.load(f)
                _global_ea_agent.best_population = saved_state.get("best_population", [])
                _global_ea_agent.best_individual = saved_state.get("best_individual", None)
                _global_ea_agent.generation_count = saved_state.get("generation_count", 0)
                _global_ea_agent.fitness_history = saved_state.get("fitness_history", [])
                print(f"Loaded EA agent state from {state_file}")
        except:
            print(f"No previous state found, starting fresh")
    
    # Get simulation from player
    simulation = player.simulation
    
    # Enable debug logging
    debug_enabled = False
    
    def debug_log(message):
        if debug_enabled:
            print(f"[EA-DEBUG] {message}")
    
    # Check for emergency situations
    if player.city.infection_level >= 3:
        # _log(f"EMERGENCY: Treating critical infection in {player.city.name}")
        return {"type": "treat", "city": player.city}
    
    # Check for hand limit
    if len(player.hand) > 7:
        discard_card = _global_ea_agent._find_best_discard(player, simulation)
        # debug_log(f"HAND LIMIT: Discarding card {getattr(discard_card, 'city_name', 'unknown')}")
        return {"type": "discard", "card": discard_card}
    
    # Update game assessment
    _global_ea_agent._update_game_assessment(player, simulation)
    
    # Decide on strategic objective
    objective = _global_ea_agent._select_strategic_objective(player, simulation)
    # debug_log(f"OBJECTIVE: {objective}")
    
    # Try to discover cure (highest priority if possible)
    cure_action = _global_ea_agent._try_discover_cure(player, simulation)
    if cure_action:
        # debug_log(f"PRIORITY: Discovering cure for {cure_action.get('color')}")
        return cure_action
    
    # Check for cooperative actions
    coop_action = _global_ea_agent._plan_cooperative_action(player, simulation)
    if coop_action:
        # debug_log(f"COOPERATION: {coop_action.get('type')}")
        return coop_action
    
    # Decide best action
    action = _global_ea_agent.decide_action(player, simulation)
    
    # Update movement history if needed
    if action.get("type") == "move":
        _global_ea_agent._update_movement_history(player, action)
        target_city = action.get("target_city")
        #debug_log(f"MOVEMENT: To {target_city.name}")
    
    # Periodically save agent state
    if random.random() < 0.1:  # 10% chance each turn
        try:
            saved_state = {
                "best_population": _global_ea_agent.best_population,
                "best_individual": _global_ea_agent.best_individual,
                "generation_count": _global_ea_agent.generation_count,
                "fitness_history": _global_ea_agent.fitness_history,
            }
            with open(state_file, "wb") as f:
                pickle.dump(saved_state, f)
                # debug_log(f"Saved agent state to {state_file}")
        except:
            debug_log(f"Failed to save agent state")
    
    return action