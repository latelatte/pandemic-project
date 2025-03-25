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