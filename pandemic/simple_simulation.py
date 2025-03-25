#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import json
import yaml
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

### ============================================================
###  Data Structures for Pandemic
### ============================================================

class City:
    """
    都市クラス：名前、感染度、隣接リスト、研究所の有無などを管理
    """
    def __init__(self, name):
        self.name = name
        self.infection_level = 0
        self.neighbours = []
        self.has_research_station = False
        self.simulation = None  # 後でPandemicSimulationインスタンスを代入

    def add_neighbour(self, other_city):
        """
        無向グラフなので、両方向でつながる
        """
        if other_city not in self.neighbours:
            self.neighbours.append(other_city)
        if self not in other_city.neighbours:
            other_city.neighbours.append(self)

    def increase_infection(self, n=1):
        """
        感染度をnだけ増加
        """
        self.infection_level += n
        # オーバーフローを避けるために最大値を仮に5に
        # (実際のパンデミックでは3を超えるとアウトブレイク扱いだが)
        if self.infection_level > 5:
            self.infection_level = 5

    def treat_infection(self, n=1):
        """
        感染度をnだけ減らす(治療)
        """
        if self.infection_level > 0:
            self.infection_level = max(0, self.infection_level - n)

    def build_research_station(self):
        self.has_research_station = True


class Card:
    """
    カードの種別: 
      - CityCard: 都市カード(色付き)
      - EpidemicCard: エピデミックを引き起こすカード
      - EventCard: イベントカード(今回は未使用や実装省略可能)
    """
    def __init__(self, card_type, city_name=None, color=None):
        self.card_type = card_type  # e.g. 'CITY', 'EPIDEMIC'
        self.city_name = city_name
        self.color = color

    def __repr__(self):
        if self.card_type == 'CITY':
            return f"CITY({self.color}, {self.city_name})"
        else:
            return f"{self.card_type}Card"


class Player:
    """
    プレイヤークラス：名前、現在の都市、手札、役割、行動戦略など
    """
    def __init__(self, name, strategy_func, strategy_name, role=None):
        self.name = name
        self.city = None
        self.hand = []
        self.role = role  # "Medic", "Researcher"等
        self.strategy_func = strategy_func
        self.strategy_name = strategy_name
        self.simulation = None

        # 1ターンに最大4アクションなどの管理用
        self.actions_remaining = 4

    def set_city(self, city):
        self.city = city

    def move_to(self, city):
        self.city = city
        print(f"[MOVE] {self.name} moved to {city.name} ({self.strategy_name})")

    def draw_card(self, card):
        self.hand.append(card)

    def discard_card(self, card):
        if card in self.hand:
            self.hand.remove(card)

    def strategy(self):
        """
        実際の行動戦略: 引数として self(=Player) を受け取る関数を呼ぶ
        """
        self.strategy_func(self)

    def perform_turn(self):
        """
        1ターンで最大4アクション行う処理をまとめる
        """
        self.actions_remaining = 4
        # ここで strategy() を呼び出して自前で4回行動してもよいし、
        # or 1アクションずつ繰り返す実装でもよい
        self.strategy()


class PandemicSimulation:
    """
    メインのパンデミックシミュレーションクラス
    - 都市
    - プレイヤー
    - デッキ: PlayerDeck, InfectionDeck
    - アウトブレイク数, 疫病の数など
    """
    def __init__(self, *strategies, max_infection_level=3, outbreak_limit=8):
        # 都市データ初期化
        self.cities = PandemicSimulation.create_cities()
        for city in self.cities:
            city.simulation = self

        # カードデッキ
        self.player_deck = self._create_player_deck()
        random.shuffle(self.player_deck)
        self.infection_deck = self._create_infection_deck()
        random.shuffle(self.infection_deck)

        # プレイヤー生成
        self.players = []
        for i, (func, name) in enumerate(strategies):
            p = Player(f"Player{i+1}", func, name)
            p.simulation = self
            self.players.append(p)

        # プレイヤーをランダムな都市へ
        for p in self.players:
            p.set_city(random.choice(self.cities))

        # 初期感染処理
        self.initial_infection()

        self.max_infection_level = max_infection_level
        self.outbreak_limit = outbreak_limit
        self.outbreak_count = 0
        self.turn_count = 0
        self.game_over = False

    @staticmethod
    def create_cities():
        # デモ用に6都市 + 適当な隣接関係
        city_names = ["Atlanta", "Chicago", "London", "Paris", "Beijing", "Tokyo"]
        city_objects = [City(n) for n in city_names]

        # 適当に接続 (全通に近いが例示)
        connections = [
            ("Atlanta", "Chicago"),
            ("Chicago", "London"),
            ("London", "Paris"),
            ("Paris", "Beijing"),
            ("Beijing", "Tokyo"),
            ("Atlanta", "Tokyo"),  # Example cross-connection
        ]
        city_dict = {c.name: c for c in city_objects}
        for c1, c2 in connections:
            city_dict[c1].add_neighbour(city_dict[c2])

        return city_objects

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
                 "role": p.role,
                 "strategy": p.strategy_name,
                 "actions_taken": [],  # 実際のアクション履歴を追加するべき
                 "hand": [str(card) for card in p.hand]}
                for p in self.players
            ],
            "outbreak_count": self.outbreak_count,
            "game_over": self.game_over,
            "win": all(c.infection_level == 0 for c in self.cities)
        }
        return data

###
### Dummy Agents for demonstration
###

def dummy_rl_agent(player: Player):
    """
    シンプルなRLダミー: 
     - 感染度が高い隣接都市に行き treatする
    """
    if player.actions_remaining <= 0:
        return
    # 1アクション程度だけ
    dangerous_city = max(player.city.neighbours, key=lambda c: c.infection_level, default=player.city)
    if dangerous_city.infection_level > player.city.infection_level:
        player.move_to(dangerous_city)
    else:
        # Move nowhere if no city is more infected
        pass
    player.city.treat_infection(1)
    player.actions_remaining -= 1


def dummy_genetic_agent(player: Player):
    """
    遺伝的アルゴリズムで学習したと想定するダミー方針
    ここでは単にランダムに行動する
    """
    if player.actions_remaining <= 0:
        return
    possible_moves = player.city.neighbours + [player.city]
    target = random.choice(possible_moves)
    player.move_to(target)
    # 50%で治療
    if random.random() < 0.5:
        player.city.treat_infection(1)
    player.actions_remaining -= 1


def dummy_mcts_agent(player: Player):
    """
    MCTSで導出したダミー方針（シンプルなロールアウトの代替）
    """
    if player.actions_remaining <= 0:
        return
    # まず現在地を治療
    if player.city.infection_level > 0:
        player.city.treat_infection(1)
        player.actions_remaining -= 1
    else:
        # Move to random neighbor
        if player.city.neighbours:
            next_c = random.choice(player.city.neighbours)
            player.move_to(next_c)
            player.actions_remaining -= 1


### 例: SimulationRunnerの拡張
class SimulationRunner:
    def __init__(self, n_episodes=10, log_dir="./logs"):
        self.n_episodes = n_episodes
        self.wins = 0
        self.losses = 0
        # TensorBoard用のWriter
        self.writer = SummaryWriter(log_dir)
        # 詳細な評価メトリクスを追加
        self.metrics = {
            'total_turns': [],
            'outbreak_counts': [],
            'treatment_counts': {},
            'win_rates': {},
            'avg_infection_level': [],
            'time_per_move': {},
            'resource_usage': {}
        }

    def run_experiments(self, strategies):
        """
        strategies: [(func, func_name), (func, func_name), ...]
        複数プレイヤー分指定
        """
        # 各戦略の性能計測を初期化
        for _, name in strategies:
            self.metrics['treatment_counts'][name] = 0
            self.metrics['win_rates'][name] = 0
            self.metrics['time_per_move'][name] = []
        
        for ep in range(self.n_episodes):
            sim = PandemicSimulation(*strategies)
            
            # 実行時間の計測開始
            start_time = time.time()
            
            # 各エージェントの行動時間を記録するためのモニター追加
            for p in sim.players:
                original_strategy = p.strategy
                strategy_name = p.strategy_name
                
                def timed_strategy(player=p, original=original_strategy, name=strategy_name):
                    move_start = time.time()
                    original(player)
                    move_end = time.time()
                    self.metrics['time_per_move'][name].append(move_end - move_start)
                
                p.strategy = timed_strategy
            
            sim.run_game()
            
            # ゲーム終了時の詳細メトリクス収集
            self.metrics['total_turns'].append(sim.turn_count)
            self.metrics['outbreak_counts'].append(sim.outbreak_count)
            self.metrics['avg_infection_level'].append(
                sum(c.infection_level for c in sim.cities) / len(sim.cities)
            )
            
            # エージェント別の治療回数集計
            for p in sim.players:
                # 実際の実装では、各エージェントのアクション記録が必要
                # ここではサンプルとして
                self.metrics['treatment_counts'][p.strategy_name] += 1
            
            # 勝敗判定と記録
            if all(c.infection_level == 0 for c in sim.cities):
                print(f"Episode {ep+1}: WIN.")
                self.wins += 1
                # 各エージェントが勝利に貢献
                for p in sim.players:
                    self.metrics['win_rates'][p.strategy_name] += 1/len(sim.players)
            else:
                print(f"Episode {ep+1}: LOSE.")
                self.losses += 1
            
            # 実行時間の計測終了
            end_time = time.time()
            print(f"Episode took {end_time - start_time:.2f} seconds")
            
            # ログ保存
            self.save_episode_log(sim, ep)

            # TensorBoardにメトリクスを記録
            self.writer.add_scalar('Metrics/Turns', sim.turn_count, ep)
            self.writer.add_scalar('Metrics/Outbreaks', sim.outbreak_count, ep)
            self.writer.add_scalar('Metrics/AvgInfection', 
                                  sum(c.infection_level for c in sim.cities) / len(sim.cities), ep)
            
            # エージェント別のメトリクス
            for p in sim.players:
                self.writer.add_scalar(f'Agent/{p.strategy_name}/MoveTimes', 
                                     np.mean(self.metrics['time_per_move'][p.strategy_name]), ep)
            
            # 勝敗をTensorBoardに記録
            win_value = 1 if all(c.infection_level == 0 for c in sim.cities) else 0
            self.writer.add_scalar('GameResult/Win', win_value, ep)
        
        # 実験終了時にTensorBoard用のサマリー統計を記録
        self.writer.add_scalar('Summary/WinRate', 100.0 * self.wins / self.n_episodes, 0)
        self.writer.add_scalar('Summary/AvgTurns', np.mean(self.metrics['total_turns']), 0)
        
        # TensorBoardのリソースを解放
        self.writer.close()

    def print_summary(self):
        """詳細な結果サマリーを出力"""
        total = self.wins + self.losses
        wrate = 100.0 * self.wins / max(1, total)
        print(f"\n===RESULT SUMMARY===")
        print(f"Wins={self.wins}, Losses={self.losses}, Rate={wrate:.2f}%")
        print(f"Average turns: {sum(self.metrics['total_turns'])/len(self.metrics['total_turns']):.2f}")
        print(f"Average outbreaks: {sum(self.metrics['outbreak_counts'])/len(self.metrics['outbreak_counts']):.2f}")
        
        print("\n===AGENT PERFORMANCE===")
        for name in self.metrics['time_per_move']:
            avg_time = sum(self.metrics['time_per_move'][name])/len(self.metrics['time_per_move'][name])
            win_contrib = self.metrics['win_rates'][name]/self.n_episodes * 100
            print(f"{name}: Avg time per move: {avg_time*1000:.2f}ms, Win contribution: {win_contrib:.2f}%")
    
    def save_episode_log(self, sim, episode_num):
        """各エピソードのログを保存"""
        log_data = sim.get_game_log()
        log_data['episode'] = episode_num
        
        # JSONファイルに保存
        with open(f"episode_{episode_num}_log.json", 'w') as f:
            json.dump(log_data, f, indent=2)

### メイン関数例
if __name__ == "__main__":

    # yamlとか別ファイルから読み込んでもいい
    # ここでは3エージェントで遊ぶ想定
    strategies = [
        (dummy_rl_agent, "Dummy-RL"),
        (dummy_genetic_agent, "Dummy-GA"),
        (dummy_mcts_agent, "Dummy-MCTS"),
    ]

    runner = SimulationRunner(n_episodes=5)
    runner.run_experiments(strategies)

    print("Done.")
