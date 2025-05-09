import random
from copy import deepcopy

class BaseAgent:
    """Base class for all agents with common functionality"""
    def __init__(self, name):
        self.name = name
        self.actions = []

    def decide_action(self, player, simulation):
        raise NotImplementedError
        
    def record_action(self, action_type, details):
        self.actions.append({
            "type": action_type,
            "details": details
        })
        
    def get_possible_actions(self, player, simulation):
        """
        Get all valid actions for the current player.
        Common implementation for all agents.
        
        Args:
            player: Current player
            simulation: Current simulation state
            
        Returns:
            List of valid action dictionaries
        """
        actions = []
        
        # 1. Standard Move (to adjacent cities)
        for neighbor in player.city.neighbours:
            actions.append({
                "type": "move",
                "target_city": neighbor,
                "method": "standard"
            })
        
        # 2. Direct Flight (discard a city card to move to that city)
        for card in player.hand:
            if hasattr(card, 'city') and card.city:
                # Skip if it's the current city
                if card.city != player.city:
                    actions.append({
                        "type": "move",
                        "target_city": card.city,
                        "method": "direct_flight",
                        "card": card
                    })
            # Also check city_name attribute for compatibility
            elif hasattr(card, 'city_name'):
                for city in simulation.cities:
                    if city.name == card.city_name and city != player.city:
                        actions.append({
                            "type": "move",
                            "target_city": city,
                            "method": "direct_flight",
                            "card": card
                        })
        
        # 3. Charter Flight (discard current city card to move anywhere)
        current_city_card = next((card for card in player.hand 
                                if (hasattr(card, 'city') and card.city == player.city) or
                                (hasattr(card, 'city_name') and card.city_name == player.city.name)), None)
        if current_city_card:
            for city in simulation.cities:
                if city != player.city:
                    actions.append({
                        "type": "move",
                        "target_city": city,
                        "method": "charter_flight",
                        "card": current_city_card
                    })
        
        # 4. Shuttle Flight (move between research stations)
        if player.city.has_research_station:
            for city in simulation.cities:
                if city != player.city and city.has_research_station:
                    actions.append({
                        "type": "move",
                        "target_city": city,
                        "method": "shuttle_flight"
                    })
        
        # 5. Treat Disease
        if player.city.infection_level > 0:
            # For each disease color in the city
            disease_colors = set()
            for color, level in player.city.disease_cubes.items():
                if level > 0:
                    disease_colors.add(color)
                    
            for color in disease_colors:
                actions.append({
                    "type": "treat",
                    "city": player.city,
                    "color": color
                })
        
        # 6. Build Research Station
        if not player.city.has_research_station:
            # Check if player has the city card or is an Operations Expert
            has_city_card = any((hasattr(card, 'city') and card.city == player.city) or
                              (hasattr(card, 'city_name') and card.city_name == player.city.name)
                              for card in player.hand)
            is_ops_expert = hasattr(player, 'role') and getattr(player.role, 'name', None) == "Operations Expert"
            
            if has_city_card or is_ops_expert:
                actions.append({
                    "type": "build",
                    "city": player.city
                })
        
        # 7. Share Knowledge (give or take city cards)
        for other_player in simulation.players:
            if other_player != player and other_player.city == player.city:
                # Give knowledge (current player has current city card)
                for card in player.hand:
                    if (hasattr(card, 'city') and card.city == player.city) or \
                       (hasattr(card, 'city_name') and card.city_name == player.city.name):
                        actions.append({
                            "type": "share_knowledge",
                            "direction": "give",
                            "target_player": other_player,
                            "card": card
                        })
                
                # Take knowledge (other player has current city card)
                for card in other_player.hand:
                    if (hasattr(card, 'city') and card.city == player.city) or \
                       (hasattr(card, 'city_name') and card.city_name == player.city.name):
                        actions.append({
                            "type": "share_knowledge",
                            "direction": "take",
                            "target_player": other_player,
                            "card": card
                        })
        
        # 8. Discover Cure
        if player.city.has_research_station:
            # Group cards by disease color
            cards_by_color = {}
            for card in player.hand:
                if hasattr(card, 'color') and card.color and card.color != "INF":
                    color = card.color
                    if color not in cards_by_color:
                        cards_by_color[color] = []
                    cards_by_color[color].append(card)
            
            # Get required cards count (usually 5, but 4 for Scientist)
            required_cards = 4 if (hasattr(player, 'role') and 
                                hasattr(player.role, 'name') and 
                                player.role.name == "Scientist") else 5
            
            # Check if player has enough cards of the same color
            for color, cards in cards_by_color.items():
                # Check if disease is already cured
                disease_cured = False
                if hasattr(simulation, 'diseases'):
                    for disease in simulation.diseases:
                        if disease["color"] == color and disease["cured"]:
                            disease_cured = True
                            break
                
                if not disease_cured and len(cards) >= required_cards:
                    actions.append({
                        "type": "discover_cure",
                        "color": color,
                        "cards": cards[:required_cards]
                    })
        
        # 9. Pass (do nothing) - usually not optimal but sometimes necessary
        actions.append({
            "type": "pass"
        })
        
        return actions
    
    def simulate_action(self, simulation, player, action):
        """
        Simulates the effect of an action without modifying the actual game state.
        
        Args:
            simulation: Current game simulation
            player: Player performing the action
            action: Action to simulate
            
        Returns:
            New simulation state after applying the action
        """
        sim_copy = deepcopy(simulation)
        
        # Find the player copy in the simulation copy
        player_copy = None
        for p in sim_copy.players:
            if p.id == player.id:
                player_copy = p
                break
        
        if not player_copy:
            return sim_copy  # Cannot find player in copy
        
        action_type = action.get("type")
        
        if action_type == "move":
            target_city = action.get("target_city")
            method = action.get("method", "standard")
            card = action.get("card")
            
            # Find the corresponding city in the copy
            target_city_copy = None
            for city in sim_copy.cities:
                if city.name == target_city.name:
                    target_city_copy = city
                    break
                    
            if target_city_copy:
                # Move player
                player_copy.city = target_city_copy
                
                # If using a card, discard it
                if method in ["direct_flight", "charter_flight"] and card:
                    card_copy = None
                    for c in player_copy.hand:
                        if (hasattr(c, 'city') and hasattr(c.city, 'name') and c.city.name == card.city.name) or \
                           (hasattr(c, 'city_name') and c.city_name == getattr(card, 'city_name', None)):
                            card_copy = c
                            break
                            
                    if card_copy:
                        player_copy.hand.remove(card_copy)
                        sim_copy.player_discard_pile.append(card_copy)
        
        elif action_type == "treat":
            city = action.get("city")
            color = action.get("color", "Blue")  # Default to Blue if not specified
            
            # Find the corresponding city in the copy
            city_copy = None
            for c in sim_copy.cities:
                if c.name == city.name:
                    city_copy = c
                    break
                    
            if city_copy:
                # Check if disease is cured
                disease_cured = False
                for disease in sim_copy.diseases:
                    if disease["color"] == color and disease["cured"]:
                        disease_cured = True
                        break
                
                # Remove disease cubes
                if disease_cured:
                    # Remove all cubes of the color
                    city_copy.disease_cubes[color] = 0
                    city_copy.infection_level = max(0, city_copy.infection_level - 1)
                else:
                    # Remove one cube
                    if city_copy.infection_level > 0:
                        city_copy.disease_cubes[color] = max(0, city_copy.disease_cubes[color] - 1)
                        city_copy.infection_level = max(0, city_copy.infection_level - 1)
        
        elif action_type == "build":
            city = player_copy.city
            city.has_research_station = True
            
            # If not Operations Expert, discard city card
            is_ops_expert = hasattr(player_copy, 'role') and getattr(player_copy.role, 'name', '') == "Operations Expert"
            if not is_ops_expert:
                city_card = next((card for card in player_copy.hand 
                                if (hasattr(card, 'city') and card.city.name == city.name) or
                                (hasattr(card, 'city_name') and card.city_name == city.name)), None)
                if city_card:
                    player_copy.hand.remove(city_card)
                    sim_copy.player_discard_pile.append(city_card)
        
        elif action_type == "discover_cure":
            color = action.get("color")
            cards = action.get("cards")
            
            # Find disease in copy
            for disease in sim_copy.diseases:
                if disease["color"] == color:
                    disease["cured"] = True
                    break
                    
            # Discard cards
            for card in cards:
                card_to_remove = None
                for c in player_copy.hand:
                    if (hasattr(c, 'color') and c.color == card.color) or \
                       (c == card):  # Direct object comparison as fallback
                        card_to_remove = c
                        break
                
                if card_to_remove:
                    player_copy.hand.remove(card_to_remove)
                    sim_copy.player_discard_pile.append(card_to_remove)
        
        elif action_type == "share_knowledge":
            direction = action.get("direction")
            target_player_id = action.get("target_player").id
            card = action.get("card")
            
            # Find target player in copy
            target_player_copy = None
            for p in sim_copy.players:
                if p.id == target_player_id:
                    target_player_copy = p
                    break
            
            if target_player_copy and card:
                if direction == "give":
                    # Find card in source player's hand
                    card_copy = next((c for c in player_copy.hand 
                                    if (hasattr(c, 'city') and hasattr(c.city, 'name') and c.city.name == card.city.name) or 
                                    (hasattr(c, 'city_name') and c.city_name == getattr(card, 'city_name', None))), None)
                    if card_copy:
                        player_copy.hand.remove(card_copy)
                        target_player_copy.hand.append(card_copy)
                else:  # take
                    # Find card in target player's hand
                    card_copy = next((c for c in target_player_copy.hand 
                                    if (hasattr(c, 'city') and hasattr(c.city, 'name') and c.city.name == card.city.name) or
                                    (hasattr(c, 'city_name') and c.city_name == getattr(card, 'city_name', None))), None)
                    if card_copy:
                        target_player_copy.hand.remove(card_copy)
                        player_copy.hand.append(card_copy)
        
        elif action_type == "pass":
            # Do nothing
            pass
            
        return sim_copy


class RandomAgent(BaseAgent):
    """random agent"""
    def __init__(self, name="Random"):
        super().__init__(name)
    
    def decide_action(self, player, simulation):
        possible_actions = self.get_possible_actions(player, simulation)
        if not possible_actions:
            return None 
        
        chosen_action = random.choice(possible_actions)
        self.record_action(chosen_action["type"], chosen_action)
        return chosen_action


def random_agent_strategy(player):
    """Strategy function for random agent"""
    agent = RandomAgent("RandomStrategy")
    return agent.decide_action(player, player.simulation)