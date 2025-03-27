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

        self.initial_infection()

        self.max_infection_level = difficulty_settings["max_infection_level"]
        self.outbreak_limit = difficulty_settings["outbreak_limit"]
        self.outbreak_count = 0
        self.turn_count = 0
        self.game_over = False
        self.discovered_cures = []

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
                print(f"Loaded config from {filepath}")
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
        

        multiplier = max(4, len(self.players) // 2)

        citycards = []
        for city in self.cities:
            for _ in range(multiplier):
                citycards.append(Card("CITY", city_name=city.name, color=random.choice(all_colors)))


        epidemic_count = 5
        epidemic_cards = [Card("EPIDEMIC")] * epidemic_count
        
        deck = citycards + epidemic_cards
        random.shuffle(deck)
        return deck

    def _create_infection_deck(self):
        infdeck = []
        for city in self.cities:
            infdeck.append(Card("CITY", city_name=city.name, color="INF"))
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
        print("==== Game start! ====")
        while not self.game_over:
            for p in self.players:
                self.turn_count += 1
                if self.game_over:
                    break
                print(f"\n--- Turn {self.turn_count}: {p.name}'s move ---")
                p.perform_turn()

                self.draw_player_cards(p)
                self.infection_phase()

                if self.check_game_end():
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
                print(f"{player.name} drew EPIDEMIC -> Infect city severely!")
                self.epidemic()
            else:
                player.draw_card(card)
                print(f"{player.name} drew {card}")

    def epidemic(self):
        if not self.infection_deck:
            self.game_over = True
            return
        bottom_card = self.infection_deck.pop(0)
        city_ = self.find_city(bottom_card.city_name)
        print(f"Epidemic hits {city_.name}!")
        city_.increase_infection(2)


    def infection_phase(self):
        if not self.infection_deck:
            print("Infection deck empty -> game might end soon")
            self.game_over = True
            return
        card = self.infection_deck.pop()
        c = self.find_city(card.city_name)
        c.increase_infection(1)
        if c.infection_level > self.max_infection_level:
            # outbreak
            self.handle_outbreak(c)

    def handle_outbreak(self, city):
        self.outbreak_count += 1
        print(f"Outbreak at {city.name}, outbreak count: {self.outbreak_count}")
        city.infection_level = self.max_infection_level  # saturate
        for nb in city.neighbours:
            nb.increase_infection(1)
        if self.outbreak_count >= self.outbreak_limit:
            self.game_over = True
            print("Too many outbreaks -> You lose")


    def discover_cure(self, player, color):
        color_cards = [card for card in player.hand if card.color == color]
        if len(color_cards) >= 5 and player.city.has_research_station:
            for _ in range(5):
                player.discard_card(color_cards.pop())
            self.discovered_cures.append(color)
            print(f"{player.name}developed {color} cure!")
            return True
        return False

    def check_game_end(self):
        if self.is_win_condition():
            print("Victory achieved!")
            self.game_over = True
            return True
        
        if self.game_over:
            return True
        
        return False
    
    def is_win_condition(self):
        """condition for winning the game"""
        # 1: infection level is controlled
        infection_controlled = all(c.infection_level <= 2 for c in self.cities)
        
        # 2: all cures are discovered
        cure_developed = len(self.discovered_cures) >= 3
        
        return infection_controlled or cure_developed
    
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