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