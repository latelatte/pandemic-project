class City:
    """
    class for cities
    """
    def __init__(self, name):
        self.name = name
        self.infection_level = 0
        self.neighbours = []
        self.has_research_station = False
        self.simulation = None
        self.outbreak_marker = False  # for preventing multiple outbreaks
        self.disease_cubes = {}  # track disease cubes for each color

    def add_neighbour(self, other_city):
        """
        connects two cities
        """
        if other_city not in self.neighbours:
            self.neighbours.append(other_city)
        if self not in other_city.neighbours:
            other_city.neighbours.append(self)

    def increase_infection(self, amount, color="Blue"):
        """increase infection level by amount"""
        old_level = self.infection_level
        self.infection_level += amount
        
        # track disease cubes for each color
        if color not in self.disease_cubes:
            self.disease_cubes[color] = 0
        self.disease_cubes[color] += amount
        
        print(f"{self.name}: infection level {old_level} -> {self.infection_level}")
        return self.infection_level

    def treat_infection(self, n=1):
        """
        treat infection level by n
        """
        if self.infection_level > 0:
            self.infection_level = max(0, self.infection_level - n)

    def build_research_station(self):
        self.has_research_station = True

    def get_infection_level(self, color="Blue"):
        """get infection level for a specific color"""
        return self.disease_cubes.get(color, 0)
        
    def set_infection_level(self, color, level):
        self.disease_cubes[color] = level
        # apply the level to the total infection level
        self.infection_level = sum(self.disease_cubes.values())