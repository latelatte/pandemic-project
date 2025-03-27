class Role:
    """class for role in Pandemic game"""
    
    def __init__(self, name, description, abilities=None):
        self.name = name
        self.description = description
        self.abilities = abilities or {}
    
    def apply_ability(self, action_type, player, **kwargs):
        if action_type not in self.abilities:
            return False
            
        ability = self.abilities[action_type]
        
        # medic ability: can treat all disease in the city
        if action_type == "treat_disease":
            if self.name == "Medic":
                city = player.city
                cubes_removed = ability.get("cubes_removed", 1)
                if city.infection_level > 0:
                    city.infection_level = max(0, city.infection_level - cubes_removed)
                    print(f"🧪 {player.name} (Medic) skill: treat {city.name} completely")
                    return True
                    
        # Scientist ability: can discover a cure with fewer cards      
        elif action_type == "develop_cure":
            if self.name == "Scientist":
                return ability.get("cards_required", 5) 
        
        # Operations Expert ability: can build a research station without a city card
        elif action_type == "build_research_station":
            if self.name == "Operations Expert":
                if ability.get("any_card", False):
                    print(f"🏗️ {player.name} (Operations Expert) skill: can build a research station with out a city card")
                    return True
        
        # Researcher ability: can share knowledge with any player
        elif action_type == "share_knowledge":
            if self.name == "Researcher":
                if ability.get("flexible", False):
                    print(f"📚 {player.name} (Researcher) skill: can share knowledge with any player")
                    return True
        
        # Dispatcher ability: can move other players
        elif action_type == "move_player":
            if self.name == "Dispatcher":
                if ability.get("can_move_others", False):
                    target = kwargs.get("target_player")
                    if target:
                        print(f"🚁 {player.name} (Dispatcher) skill: support the move of {target.name}")
                        return True
        
        # Quarantine Specialist ability: can prevent infection in a city
        elif action_type == "prevent_infection":
            if self.name == "Quarantine Specialist":
                radius = ability.get("radius", 1)
                city = kwargs.get("city")
                if city and (city == player.city or city in player.city.neighbours):
                    print(f"🛡️ {player.name} (Quarantine Specialist) skill: prevent form infection at {city.name}")
                    return True
        
        return False