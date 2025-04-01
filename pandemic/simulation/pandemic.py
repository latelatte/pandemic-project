import random
import json
import os
from pandemic.models.city import City
from pandemic.models.card import Card
from pandemic.models.player import Player
from pandemic.models.role import Role


class PandemicSimulation:
    """
    main class for the Pandemic simulation game.
    Handles game initialization, player management, and game loop.
    """
    def __init__(self, *strategies, config_dir="./config", difficulty="normal", num_players=None):
        cities_config = self._load_config(config_dir, "cities_config.json")
        diseases_config = self._load_config(config_dir, "diseases_config.json")
        roles_config = self._load_config(config_dir, "roles_config.json")
        game_config = self._load_config(config_dir, "game_config.json")
        
        difficulty_settings = game_config["difficulty_levels"][difficulty]
        
        self.max_infection_level = difficulty_settings["max_infection_level"]
        self.outbreak_limit = difficulty_settings["outbreak_limit"]
        self.outbreak_count = 0
        self.turn_count = 0
        self.game_over = False
        self.discovered_cures = []
        self.infection_rates = [2, 2, 2, 3, 3, 4, 4] 
        self.infection_rate_index = 0
        
        # 疾病の初期化
        self.diseases = []
        disease_colors = ["Blue", "Red", "Yellow", "Black"]
        for color in disease_colors[:3]:
            self.diseases.append({
                "color": color,
                "cured": False,
                "eradicated": False
            })

        self.player_discard_pile = []
        self.infection_discard_pile = []
        
        self.infection_rate_index = difficulty_settings.get("initial_infection_rate_index", 0)

        self.actions_per_turn = game_config.get("actions_per_turn", 4)  # デフォルト値として4を設定

        self.cities = self._create_cities(cities_config)
        for city in self.cities:
            city.simulation = self

        self.roles = []
        for role_data in roles_config.get("roles", []):
            role = Role(
                name=role_data["name"],
                description=role_data["description"],
                abilities=role_data.get("abilities", {})
            )
            self.roles.append(role)

        player_count = num_players if num_players is not None else 4
        self.players = []

        available_roles = self.roles.copy()
        random.shuffle(available_roles)

        for i in range(player_count):
            strategy_index = i % len(strategies)
            func, name = strategies[strategy_index]
            
            player_name = f"{name}-{i+1}" if len(strategies) < player_count else name
            
            p = Player(player_name, func, name)
            p.simulation = self
            
            if available_roles:
                role = available_roles.pop(0)
                p.assign_role(role)
            
            self.players.append(p)

        for p in self.players:
            p.set_city(random.choice(self.cities))

        self.player_deck = self._create_player_deck()
        random.shuffle(self.player_deck)
        self.infection_deck = self._create_infection_deck()
        random.shuffle(self.infection_deck)

        # Deal initial hand to each player
        cards_per_player = difficulty_settings.get("player_cards_initial", 2)
        for p in self.players:
            for _ in range(cards_per_player):
                if self.player_deck:
                    card = self.player_deck.pop()
                    if card.card_type != "EPIDEMIC":  # Skip epidemic cards in initial deal
                        p.hand.append(card)
                    else:
                        # Put epidemic back and draw another one
                        self.player_deck.insert(0, card)
                        if self.player_deck:
                            p.hand.append(self.player_deck.pop())

        self.initial_infection()

    def _load_config(self, config_dir, filename):
        """load configuration from JSON file and merge with defaults"""
        filepath = os.path.join(config_dir, filename)
        
        defaults = {
            "game_config.json": {
                "difficulty_levels": {
                    "normal": {
                        "epidemic_cards": 5,
                        "player_cards_initial": 3,
                        "player_cards_per_turn": 2,
                        "max_infection_level": 3,
                        "outbreak_limit": 8
                    }
                },
                "actions_per_turn": 4,
                "cards_needed_for_cure": 5,
                "default_difficulty": "normal"
            },
        }
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
                # print(f"Loaded config from {filepath}")
                return config
        except FileNotFoundError:
            print(f"Warning: Config file {filepath} not found. Using defaults.")
            return defaults.get(filename, {})
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {filepath}. Using defaults.")
            return defaults.get(filename, {})

    def _create_cities(self, config):
        cities = []
        city_dict = {}
        
        for city_data in config.get("cities", []):
            city = City(city_data["name"])
            if city_data.get("initial_research_station", False):
                city.has_research_station = True
            city_dict[city.name] = city
            cities.append(city)
            
        for conn in config.get("connections", []):
            city1, city2 = conn
            if city1 in city_dict and city2 in city_dict:
                city_dict[city1].add_neighbour(city_dict[city2])
                
        return cities

    def _create_player_deck(self):
        all_colors = ["Blue", "Red", "Yellow"]
        
        # 都市数が20未満の場合、multiplierを大きくする（都市数が少ないことを補償）
        if len(self.cities) < 20:
            multiplier = 6  # 大幅に増加
        else:
            multiplier = 4
        
        citycards = []
        for city in self.cities:
            for _ in range(multiplier):
                citycards.append(Card("CITY", city_name=city.name, color=random.choice(all_colors)))

        # エピデミックカード数を調整（難易度設定から取得）
        epidemic_count = 3  # 少なくした
        epidemic_cards = [Card("EPIDEMIC")] * epidemic_count
        
        deck = citycards + epidemic_cards
        random.shuffle(deck)
        print(f"Created player deck with {len(deck)} cards ({len(citycards)} city cards, {epidemic_count} epidemics)")
        print(f"Total cities: {len(self.cities)}")  # 都市数を出力してデバッグ
        return deck

    def _create_infection_deck(self):
        """
        Create infection deck with multiple copies of each city card
        to prevent early deck depletion
        """
        infdeck = []
        
        # City count affects how many copies we need
        if len(self.cities) < 15:
            multiplier = 8  # More copies for fewer cities
        else:
            multiplier = 4  # Fewer copies for more cities
        
        for city in self.cities:
            for _ in range(multiplier):
                infdeck.append(Card("CITY", city_name=city.name, color="INF"))
        
        print(f"Created infection deck with {len(infdeck)} cards")
        return infdeck

    def initial_infection(self):
        for _ in range(3):
            if self.infection_deck:
                top_card = self.infection_deck.pop()
                city_ = self.find_city(top_card.city_name)
                city_.increase_infection(3)
                print(f"Inital infection: {city_.name} infected with 3 cubes")
        

        for _ in range(3):
            if self.infection_deck:
                top_card = self.infection_deck.pop()
                city_ = self.find_city(top_card.city_name)
                city_.increase_infection(2)
                print(f"Initial infection: {city_.name} infected with 2 cubes")
        

        for _ in range(3):
            if self.infection_deck:
                top_card = self.infection_deck.pop()
                city_ = self.find_city(top_card.city_name)
                city_.increase_infection(1)
                print(f"Initial indection: {city_.name} infected with 1 cube")

    def find_city(self, name):
        for c in self.cities:
            if c.name == name:
                return c
        return None

    def run_game(self):
        """
        actual game loop
        """
        print(f"==== Game start! ==== (Player deck: {len(self.player_deck)} cards)")
        while not self.game_over:
            for p in self.players:
                self.turn_count += 1
                if self.game_over:
                    break
                print(f"\n--- Turn {self.turn_count}: {p.name}'s move --- (Player deck: {len(self.player_deck)} cards)")
                p.perform_turn()

                self.draw_player_cards(p)
                self.infection_phase()
                
                # ターン終了時にアウトブレイクマーカーをリセット
                for city in self.cities:
                    if hasattr(city, 'outbreak_marker'):
                        city.outbreak_marker = False
                
                print(f"End of turn {self.turn_count} - Player deck has {len(self.player_deck)} cards remaining")

                if self.is_win_condition():
                    print("Victory! All diseases cured!")
                    self.game_over = True
                    break

        print("==== Game ended. ====")

    def draw_player_cards(self, player, n=2):
        for _ in range(n):
            if not self.player_deck:
                print("No more cards in player deck -> Lose!")
                self.game_over = True
                return

            card = self.player_deck.pop()
            if card.card_type == "EPIDEMIC":
                print(f"Epidemic card drawn by {player.name}! (Not added to hand)")
                self.epidemic()
            else:
                player.hand.append(card)
                print(f"{player.name} draws {card}")
                
        max_hand_size = 7
        while len(player.hand) > max_hand_size:
            print(f"{player.name} must discard down to {max_hand_size} cards")
            discard_action = player.strategy(player)
            if discard_action and discard_action.get("type") == "discard":
                card = discard_action.get("card")
                if card in player.hand:
                    player.hand.remove(card)
                    self.player_discard_pile.append(card)
                    print(f"{player.name} discarded {card}")
                else:
                    card = random.choice(player.hand)
                    player.hand.remove(card)
                    self.player_discard_pile.append(card)
                    print(f"{player.name} randomly discarded {card}")
            else:
                card = random.choice(player.hand)
                player.hand.remove(card)
                self.player_discard_pile.append(card)
                print(f"{player.name} randomly discarded {card}")

    @property
    def infection_rate(self):
        if self.infection_rate_index < len(self.infection_rates):
            return self.infection_rates[self.infection_rate_index]
        return self.infection_rates[-1]
    
    def epidemic(self):
        if not self.infection_deck:
            self.game_over = True
            return
        
        if self.infection_rate_index < len(self.infection_rates) - 1:
            self.infection_rate_index += 1
            print(f"Infection rate increased to {self.infection_rate}")
        
        bottom_card = self.infection_deck.pop(0)  # 一番下のカードを取得
        city_ = self.find_city(bottom_card.city_name)
        print(f"Epidemic hits {city_.name}!")
        city_.increase_infection(3)  

        # ここでインフェクションの捨て札をデッキに戻しています
        if self.infection_discard_pile:
            random.shuffle(self.infection_discard_pile)
            self.infection_deck.extend(self.infection_discard_pile)
            self.infection_discard_pile = []
            print("Infection discard pile shuffled and placed on top of infection deck")

    def infection_phase(self):
        cards_to_draw = self.infection_rate
        print(f"Drawing {cards_to_draw} infection cards...")
        
        for _ in range(cards_to_draw):
            if not self.infection_deck:
                self.game_over = True
                print("Infection deck empty -> Game over!")
                return
                
            card = self.infection_deck.pop()
            self.infection_discard_pile.append(card)
            c = self.find_city(card.city_name)
            

            disease_color = card.color if hasattr(card, 'color') and card.color != "INF" else "Blue"
            is_eradicated = False
            
            for disease in self.diseases:
                if disease["color"] == disease_color and disease["eradicated"]:
                    is_eradicated = True
                    
            if not is_eradicated:
                c.increase_infection(1)
                print(f"{c.name} infected with 1 cube ({disease_color})")
            else:
                print(f"Disease {disease_color} is eradicated, no infection in {c.name}")

    def handle_outbreak(self, city, disease_color="Blue"):
        self.outbreak_count += 1
        print(f"Outbreak at {city.name}, outbreak count: {self.outbreak_count}")
        city.infection_level = self.max_infection_level  
        city.outbreak_marker = True  
        
        for nb in city.neighbours:
            if not getattr(nb, 'outbreak_marker', False):
                nb.increase_infection(1)
                if nb.infection_level > self.max_infection_level:
                    self.handle_outbreak(nb, disease_color)
        
        if self.outbreak_count >= self.outbreak_limit:
            self.game_over = True
            print("Too many outbreaks -> You lose")

    def discover_cure(self, player, color):
        cards_needed = 4 if hasattr(player, 'role') and player.role.name == "Scientist" else 5
        
        color_cards = [card for card in player.hand if hasattr(card, 'color') and card.color == color]
        if len(color_cards) >= cards_needed and player.city.has_research_station:
            for i in range(cards_needed):
                card = color_cards[i]
                player.hand.remove(card)
                self.player_discard_pile.append(card)

            for disease in self.diseases:
                if disease["color"] == color:
                    disease["cured"] = True
                    print(f"Cure for {color} disease discovered!")
                    self.discovered_cures.append(color)
                    return True
        return False
    
    def treat_disease(self, player, city, color=None):
        if color is None:
            color = "Blue" 
        
        is_cured = False
        for disease in self.diseases:
            if disease["color"] == color and disease["cured"]:
                is_cured = True
                break

        if city.infection_level > 0:
            if is_cured:
                old_level = city.infection_level
                city.infection_level = 0
                print(f"{player.name} removed all {old_level} disease cubes from {city.name} ({color} is cured)")
            else:
                city.infection_level -= 1
                print(f"{player.name} removed 1 disease cube from {city.name}")
            return True
        return False

    def is_win_condition(self):
        all_diseases_cured = all(disease["cured"] for disease in self.diseases)
        
        return all_diseases_cured
    
    def show_status(self):
        print("\n--- Current City Status ---")
        for c in self.cities:
            bar = "#" * c.infection_level
            print(f"{c.name} (inf={c.infection_level}): [{bar}]")
        print(f"Outbreaks: {self.outbreak_count}\n")

    def get_game_log(self):
        data = {
            "turn": self.turn_count,
            "cities": [
                {"name": city.name,
                 "infection_level": city.infection_level,
                 "neighbors": [n.name for n in city.neighbours],
                 "research_station": city.has_research_station}
                for city in self.cities
            ],
            "players": [
                {"name": p.name,
                 "city": p.city.name if p.city else None,
                 "role": p.role.name if hasattr(p, 'role') and p.role else None,
                 "role_description": p.role.description if hasattr(p, 'role') and p.role else None,
                 "strategy": p.strategy_name,
                 "actions_taken": [],
                 "hand": [str(card) for card in p.hand]}
                for p in self.players
            ],
            "outbreak_count": self.outbreak_count,
            "game_over": self.game_over,
            "win": self.is_win_condition()
        }
        return data