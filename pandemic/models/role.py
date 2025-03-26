class Role:
    """パンデミックにおけるプレイヤーの役割を表現するクラス"""
    
    def __init__(self, name, description, abilities=None):
        self.name = name
        self.description = description
        self.abilities = abilities or {}
    
    def apply_ability(self, action_type, player, **kwargs):
        """アクションに役割の特殊能力を適用"""
        if action_type not in self.abilities:
            return False
            
        ability = self.abilities[action_type]
        
        # Medicの能力: 治療効果が2倍
        if action_type == "treat_disease":
            if self.name == "Medic":
                city = player.city
                cubes_removed = ability.get("cubes_removed", 1)
                if city.infection_level > 0:
                    city.infection_level = max(0, city.infection_level - cubes_removed)
                    print(f"🧪 {player.name} (Medic) 特殊能力: {city.name}の感染を{cubes_removed}段階治療")
                    return True
                    
        # Scientistの能力: 治療薬開発に必要なカード数減少        
        elif action_type == "develop_cure":
            if self.name == "Scientist":
                return ability.get("cards_required", 5)  # 返り値は必要カード枚数
        
        # Operations Expertの能力: 任意のカードで研究所を建設可能
        elif action_type == "build_research_station":
            if self.name == "Operations Expert":
                if ability.get("any_card", False):
                    print(f"🏗️ {player.name} (Operations Expert) 特殊能力: 任意のカードで研究所を建設")
                    return True
        
        # Researcherの能力: 知識共有の制約緩和
        elif action_type == "share_knowledge":
            if self.name == "Researcher":
                if ability.get("flexible", False):
                    print(f"📚 {player.name} (Researcher) 特殊能力: 柔軟な知識共有")
                    return True
        
        # Dispatcherの能力: 他プレイヤーの移動を補助
        elif action_type == "move_player":
            if self.name == "Dispatcher":
                if ability.get("can_move_others", False):
                    target = kwargs.get("target_player")
                    if target:
                        print(f"🚁 {player.name} (Dispatcher) 特殊能力: {target.name}の移動を支援")
                        return True
        
        # Quarantine Specialistの能力: 感染防止
        elif action_type == "prevent_infection":
            if self.name == "Quarantine Specialist":
                radius = ability.get("radius", 1)
                city = kwargs.get("city")
                if city and (city == player.city or city in player.city.neighbours):
                    print(f"🛡️ {player.name} (Quarantine Specialist) 特殊能力: {city.name}の感染を防止")
                    return True
        
        return False