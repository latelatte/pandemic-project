import random
import json
import os
from pandemic.models.city import City
from pandemic.models.card import Card
from pandemic.models.player import Player
from pandemic.models.role import Role


class PandemicSimulation:
    """
    メインのパンデミックシミュレーションクラス
    - 都市
    - プレイヤー
    - デッキ: PlayerDeck, InfectionDeck
    - アウトブレイク数, 疫病の数など
    """
    def __init__(self, *strategies, config_dir="./config", difficulty="normal", num_players=None):
        # 設定ファイルの読み込み
        cities_config = self._load_config(config_dir, "cities_config.json")
        diseases_config = self._load_config(config_dir, "diseases_config.json")
        roles_config = self._load_config(config_dir, "roles_config.json")
        game_config = self._load_config(config_dir, "game_config.json")
        
        # ゲーム難易度の設定
        difficulty_settings = game_config["difficulty_levels"][difficulty]
        
        # 都市作成
        self.cities = self._create_cities(cities_config)
        for city in self.cities:
            city.simulation = self

        # カードデッキ
        self.player_deck = self._create_player_deck()
        random.shuffle(self.player_deck)
        self.infection_deck = self._create_infection_deck()
        random.shuffle(self.infection_deck)

        # 役割の読み込みと作成
        self.roles = []
        for role_data in roles_config.get("roles", []):
            role = Role(
                name=role_data["name"],
                description=role_data["description"],
                abilities=role_data.get("abilities", {})
            )
            self.roles.append(role)

        # プレイヤー生成と役割割り当て
        player_count = num_players if num_players is not None else 4  # デフォルト4人
        self.players = []

        # 役割をシャッフル
        available_roles = self.roles.copy()
        random.shuffle(available_roles)

        # 全エージェントに役割を割り当てる
        for i in range(player_count):
            strategy_index = i % len(strategies)  # 循環的に割り当て
            func, name = strategies[strategy_index]
            
            # プレイヤー番号を名前に追加
            player_name = f"{name}-{i+1}" if len(strategies) < player_count else name
            
            p = Player(player_name, func, name)
            p.simulation = self
            
            # 役割の割り当て（利用可能な役割から）
            if available_roles:
                role = available_roles.pop(0)
                p.assign_role(role)
            
            self.players.append(p)

        # プレイヤーをランダムな都市へ
        for p in self.players:
            p.set_city(random.choice(self.cities))

        # 初期感染処理
        self.initial_infection()

        self.max_infection_level = difficulty_settings["max_infection_level"]
        self.outbreak_limit = difficulty_settings["outbreak_limit"]
        self.outbreak_count = 0
        self.turn_count = 0
        self.game_over = False

    def _load_config(self, config_dir, filename):
        """設定ファイルを読み込み、デフォルト値とマージする"""
        filepath = os.path.join(config_dir, filename)
        
        # デフォルト設定（ファイルごとに異なる）
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
            # 必要に応じて他の設定ファイルのデフォルト値も追加
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
        """設定に基づいて都市とネットワークを作成"""
        cities = []
        city_dict = {}
        
        # 都市作成
        for city_data in config.get("cities", []):
            city = City(city_data["name"])
            if city_data.get("initial_research_station", False):
                city.has_research_station = True
            city_dict[city.name] = city
            cities.append(city)
            
        # 接続関係設定
        for conn in config.get("connections", []):
            city1, city2 = conn
            if city1 in city_dict and city2 in city_dict:
                city_dict[city1].add_neighbour(city_dict[city2])
                
        return cities

    def _create_player_deck(self):
        # 色とか適当につける
        citycards = []
        all_colors = ["Blue", "Red", "Yellow"]
        for city in self.cities:
            color = random.choice(all_colors)
            citycards.append(Card("CITY", city_name=city.name, color=color))

        # エピデミックカード 2枚くらい
        epidemic_cards = [Card("EPIDEMIC")] * 2
        deck = citycards + epidemic_cards
        return deck

    def _create_infection_deck(self):
        # すべてCITYカード扱いだが、実際にはinfection card
        infdeck = []
        for city in self.cities:
            infdeck.append(Card("CITY", city_name=city.name, color="INF"))
        return infdeck

    def initial_infection(self):
        # 3枚引いて、それぞれ感染度上げるなど
        for _ in range(3):
            if self.infection_deck:
                top_card = self.infection_deck.pop()
                city_ = self.find_city(top_card.city_name)
                city_.increase_infection(1)

    def find_city(self, name):
        for c in self.cities:
            if c.name == name:
                return c
        return None

    def run_game(self):
        """
        実際のメインループ: 全員のターンを回し、勝敗判定まで
        """
        print("==== Game start! ====")
        while not self.game_over:
            for p in self.players:
                self.turn_count += 1
                if self.game_over:
                    break
                print(f"\n--- Turn {self.turn_count}: {p.name}'s move ---")
                p.perform_turn()

                # ターン後にプレイヤーカード2枚ドロー(省略的実装)
                self.draw_player_cards(p)
                # 感染フェーズ
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
        bottom_card = self.infection_deck.pop(0)  # bottom
        city_ = self.find_city(bottom_card.city_name)
        print(f"Epidemic hits {city_.name}!")
        city_.increase_infection(2)
        # discard pileシャッフル等は省略

    def infection_phase(self):
        # 今は毎ターン 1枚だけ引いて infection
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

    def check_game_end(self):
        # 勝利判定(本来は4色の治療薬が完成か？)
        # ここでは簡易的に city全部 infection_level == 0 ならwin
        if all(c.infection_level == 0 for c in self.cities):
            print("All cities are infection-free -> Win!")
            self.game_over = True
            return True
        return self.game_over

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
            "win": all(c.infection_level == 0 for c in self.cities)
        }
        return data