class Card:
    """
    Represents a card in the game.
    Cards can be of different types, such as CITY or EPIDEMIC.
    Each card may have additional attributes like city name and color.
    Attributes:
        card_type (str): The type of the card (e.g., 'CITY', 'EPIDEMIC').
        city_name (str): The name of the city associated with the card, if applicable.
        color (str): The color associated with the card, if applicable.
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