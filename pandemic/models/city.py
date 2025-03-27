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

    def add_neighbour(self, other_city):
        """
        connects two cities
        """
        if other_city not in self.neighbours:
            self.neighbours.append(other_city)
        if self not in other_city.neighbours:
            other_city.neighbours.append(self)

    def increase_infection(self, n=1):
        """
        infection level increases by n
        """
        self.infection_level += n
        if self.infection_level > 3:
            self.infection_level = 3

    def treat_infection(self, n=1):
        """
        treat infection level by n
        """
        if self.infection_level > 0:
            self.infection_level = max(0, self.infection_level - n)

    def build_research_station(self):
        self.has_research_station = True