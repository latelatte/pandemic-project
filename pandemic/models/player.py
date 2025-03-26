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

    def available_actions(self):
        """利用可能なアクションのリストを返す"""
        actions = []
        
        # 1. 移動アクション（複数タイプ）
        actions.extend(self._get_movement_actions())
        
        # 2. 治療アクション
        if self.city.infection_level > 0:
            for disease_color in self.city.infections:
                if self.city.infections[disease_color] > 0:
                    actions.append({"type": "treat", "city": self.city, "color": disease_color})
        
        # 3. 知識の共有
        actions.extend(self._get_share_knowledge_actions())
        
        # 4. 研究所の建設
        if not self.city.has_research_station and self._can_build_research_station():
            actions.append({"type": "build", "city": self.city})
        
        # 5. 治療薬の開発
        if self._can_discover_cure():
            for color in ["Blue", "Red", "Yellow", "Black"]:
                if self._has_enough_cards_for_cure(color):
                    actions.append({"type": "cure", "color": color})
        
        return actions