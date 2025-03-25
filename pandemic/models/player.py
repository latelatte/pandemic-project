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