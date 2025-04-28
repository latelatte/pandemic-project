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

    def increase_infection(self, amount=1, color="Blue"):
        old_level = self.infection_level
        self.infection_level += amount
        print(f"{self.name}: infection level {old_level} -> {self.infection_level}")
        
        if hasattr(self, 'simulation') and self.simulation:
            max_level = getattr(self.simulation, 'max_infection_level', 3)  # default 3
            
            if self.infection_level > max_level:
                if not hasattr(self, 'outbreak_marker') or not self.outbreak_marker:
                    self.outbreak_marker = True
                    if hasattr(self.simulation, 'handle_outbreak'):
                        self.simulation.handle_outbreak(self, color)

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