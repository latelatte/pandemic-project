class Player:
    """
    class for player
    """
    # クラス変数としてカウンタを追加
    _id_counter = 0
    
    def __init__(self, name, strategy_func, strategy_name, role=None):
        # 一意のIDを割り当て
        self.id = Player._id_counter
        Player._id_counter += 1
        
        self.name = name
        self.city = None
        self.hand = []
        self.role = role
        self.strategy_func = strategy_func
        self.strategy_name = strategy_name
        self.simulation = None

        self.actions_remaining = 4

    def set_city(self, city):
        self.city = city

    def move_to(self, city):
        self.city = city
        print(f"[MOVE] {self.name} moved to {city.name} ({self.strategy_name})")
        return True

    def draw_card(self, card):
        self.hand.append(card)

    def discard_card(self, card):
        if card in self.hand:
            self.hand.remove(card)

    def strategy(self):
        """
        actual strategy function
        """
        self.strategy_func(self)

    def perform_turn(self):
        actions_remaining = 4
        
        while actions_remaining > 0:
            action = self.strategy_func(self) if self.strategy_func else None
            
            if not action:
                print(f"{self.name} has skipped.")
                actions_remaining -= 1
                continue
            
            success = False
            
            if action.get("type") == "move":
                target = action.get("target")
                if target:
                    success = self.move_to(target)
            
            elif action.get("type") == "treat":
                target = action.get("target") or self.city
                if target and target.infection_level > 0:
                    success = self.treat_disease(target)
            
            actions_remaining -= 1

    def available_actions(self):
        actions = []
        # 1 movement actions
        actions.extend(self._get_movement_actions())
        
        # 2 treatment actions
        if self.city.infection_level > 0:
            for disease_color in self.city.infections:
                if self.city.infections[disease_color] > 0:
                    actions.append({"type": "treat", "city": self.city, "color": disease_color})
        
        # 3 share knowledge actions
        actions.extend(self._get_share_knowledge_actions())
        
        # 4 build research station
        if not self.city.has_research_station and self._can_build_research_station():
            actions.append({"type": "build", "city": self.city})
        
        # 5 discover cure actions
        if self._can_discover_cure():
            for color in ["Blue", "Red", "Yellow", "Black"]:
                if self._has_enough_cards_for_cure(color):
                    actions.append({"type": "cure", "color": color})
        
        return actions

    def assign_role(self, role):
        self.role = role
        self.role_name = role.name
        print(f"{self.name} is roled as {role.name}.: {role.description}")

    def use_ability(self, action_type, **kwargs):
        if hasattr(self, 'role') and self.role:
            return self.role.apply_ability(action_type, self, **kwargs)
        return False

    def treat_disease(self, city=None):
        city = city or self.city
        if city.infection_level <= 0:
            return False
        
        if hasattr(self, 'role') and self.role and self.role.name == "Medic":
            print(f"🧪 {self.name} (Medic) ability: completely treat {city.name} infection.")
            city.infection_level = 0
            return True
        
        city.infection_level -= 1
        print(f"{self.name} treated {city.name}")
        return True

    def build_research_station(self):
        if self.use_ability("build_research_station"):
            self.city.has_research_station = True
            return True
        
        city_card = None
        for card in self.hand:
            if card.type == "city" and card.city_name == self.city.name:
                city_card = card
                break
        
        if city_card:
            self.discard_card(city_card)
            self.city.has_research_station = True
            print(f"{self.name} built research station at {self.city.name}.")
            return True
        
        return False