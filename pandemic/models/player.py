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
        """都市間の移動。成功したらTrueを返す"""
        self.city = city
        # ログ出力はここで行う（perform_turnでは行わない）
        print(f"[MOVE] {self.name} moved to {city.name} ({self.strategy_name})")
        return True  # 移動が成功したことを示すため、Trueを返す

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
        actions_remaining = 4  # 標準的なPandemicでは4アクション
        
        while actions_remaining > 0:
            action = self.strategy_func(self) if self.strategy_func else None
            
            if not action:
                print(f"{self.name}はアクションをスキップしました")
                actions_remaining -= 1
                continue
            
            success = False  # アクション実行の成否
            
            if action.get("type") == "move":
                target = action.get("target")
                if target:
                    success = self.move_to(target)  # 移動の実行
            
            elif action.get("type") == "treat":
                target = action.get("target") or self.city
                if target and target.infection_level > 0:
                    success = self.treat_disease(target)  # 治療の実行
            
            # アクションの実行結果に関わらずカウント
            actions_remaining -= 1

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

    def assign_role(self, role):
        """役割を割り当てる"""
        self.role = role
        self.role_name = role.name
        print(f"{self.name}に役割「{role.name}」が割り当てられました: {role.description}")

    def use_ability(self, action_type, **kwargs):
        """特殊能力を使用"""
        if hasattr(self, 'role') and self.role:
            return self.role.apply_ability(action_type, self, **kwargs)
        return False

    def treat_disease(self, city=None):
        """治療アクション - 役割能力を活用"""
        city = city or self.city
        if city.infection_level <= 0:
            return False
        
        # 役割能力のチェック
        if hasattr(self, 'role') and self.role and self.role.name == "Medic":
            # Medicの能力を使って全キューブ除去
            print(f"🧪 {self.name} (Medic) 特殊能力: {city.name}の感染を完全に治療")
            city.infection_level = 0
            return True
        
        # 通常の治療
        city.infection_level -= 1
        print(f"{self.name}が{city.name}の感染を1段階治療しました")
        return True

    def build_research_station(self):
        """研究所建設 - 役職能力対応"""
        # Operations Expertの特殊能力を試す
        if self.use_ability("build_research_station"):
            self.city.has_research_station = True
            return True
        
        # 通常の建設（都市カードが必要）
        city_card = None
        for card in self.hand:
            if card.type == "city" and card.city_name == self.city.name:
                city_card = card
                break
        
        if city_card:
            self.discard_card(city_card)
            self.city.has_research_station = True
            print(f"{self.name}が{self.city.name}に研究所を建設しました")
            return True
        
        return False