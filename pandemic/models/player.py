import random
class Player:
    """
    class for player
    """
    _id_counter = 0
    
    def __init__(self, name, strategy_func, strategy_name, role=None):
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

    def strategy(self, player=None):
        """
        actual strategy function
        Returns the action to perform
        """
        if player is None:
            player = self
        return self.strategy_func(player) 

    def perform_turn(self):
        remaining_actions = self.simulation.actions_per_turn
        print(f"--- {self.name}'s turn starting with {len(self.hand)} cards in hand ---")
        
        used_cards = set() 
        while remaining_actions > 0:
            available_hand = [card for card in self.hand if id(card) not in [id(c) for c in used_cards]]
            
            actual_hand = self.hand.copy()
            self.hand = available_hand
            
            action = self.strategy(self)
            
            self.hand = actual_hand
            
            if not action:
                break
            
            self._execute_action(action)
            remaining_actions -= 1
            
            if action["type"] == "move" and action.get("method") in ["direct_flight", "charter_flight"]:
                card = action.get("card")
                if card:
                    if card in self.hand: 
                        used_cards.add(card)
                        self.hand.remove(card)
                        self.simulation.player_discard_pile.append(card)
                        print(f"{self.name} discarded {card} for {action.get('method')}")
                    else:
                        print(f"ERROR: Card {card} not in hand!")
                        action["method"] = "standard" 
            

            elif action["type"] == "discover_cure":
                cards = action.get("cards", [])
                valid_cards = [c for c in cards if c in self.hand]
                if len(valid_cards) >= 4: 
                    for card in valid_cards[:5]: 
                        if card in self.hand:
                            used_cards.add(card)
                            self.hand.remove(card)
                            self.simulation.player_discard_pile.append(card)
                    print(f"{self.name} used {len(valid_cards)} cards for cure discovery")
                else:
                    print(f"ERROR: Not enough valid cards for cure discovery")
                    action = {"type": "pass"} 
            

            self._execute_action(action)
            remaining_actions -= 1


    def _execute_action(self, action):
        """
        アクションを適切なメソッド呼び出しに委譲する
        """
        action_type = action.get("type")
        
        if action_type == "move":
            target_city = action.get("target_city")
            if target_city:
                self.move_to(target_city)
        
        elif action_type == "treat":
            city = action.get("city", self.city)
            color = action.get("color", "Blue")
            self.simulation.treat_disease(self, city, color)
        
        elif action_type == "build":
            card = action.get("card")
            self.build_research_station(card)
        
        elif action_type == "discover_cure":
            color = action.get("color")
            self.simulation.discover_cure(self, color)
        
        elif action_type == "share_knowledge":
            direction = action.get("direction")
            target_player = action.get("target_player")
            card = action.get("card")
            
            if card and target_player:
                if direction == "give":
                    target_player.hand.append(card)
                    print(f"{self.name} gave {card} to {target_player.name}")
                else:  # take
                    self.hand.append(card)
                    print(f"{self.name} took {card} from {target_player.name}")
        
        elif action_type == "pass":
            print(f"{self.name} passes their action")

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

    def build_research_station(self, card=None):
        """
        Build a research station in the current city
        """
        # すでに研究所がある場合は何もしない
        if self.city.has_research_station:
            print(f"{self.city.name} already has a research station")
            return False
        
        # カードが指定されていなければ手札から適切なカードを探す
        if card is None:
            for c in self.hand:
                if (hasattr(c, 'city_name') and c.city_name == self.city.name) or \
                   (hasattr(c, 'city') and hasattr(c.city, 'name') and c.city.name == self.city.name):
                    card = c
                    break
        
        # Operations Expert役職の確認
        is_ops_expert = False
        if hasattr(self, 'role') and hasattr(self.role, 'name'):
            is_ops_expert = self.role.name == "Operations Expert"
        
        # カードを使用するか、OpsExpertの能力で建設
        if card:
            # カード属性の存在チェック
            if ((hasattr(card, 'city_name') and card.city_name == self.city.name) or
               (hasattr(card, 'city') and hasattr(card.city, 'name') and card.city.name == self.city.name)):
                # カードを捨て札に
                self.hand.remove(card)
                self.simulation.player_discard_pile.append(card)
                self.city.has_research_station = True
                print(f"{self.name} built a research station at {self.city.name}")
                return True
            return False
        elif is_ops_expert:
            # OpsExpertは市のカードなしで建設可能
            self.city.has_research_station = True
            print(f"{self.name} (Operations Expert) built a research station at {self.city.name}")
            return True
        
        print(f"Cannot build research station at {self.city.name}: no matching city card")
        return False