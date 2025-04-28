# import pygame
# import sys
# import json
# import math

# class PandemicViewer:
#     def __init__(self, width=1024, height=768):
#         pygame.init()
#         self.screen = pygame.display.set_mode((width, height))
#         pygame.display.set_caption("Pandemic Simulation Viewer")
        
#         self.width = width
#         self.height = height
        
#         self.colors = {
#             "background": (30, 30, 50),
#             "city_normal": (200, 200, 200),
#             "infection_low": (220, 150, 150),
#             "infection_med": (240, 100, 100),
#             "infection_high": (255, 50, 50),
#             "research": (100, 150, 255),
#             "player": (255, 255, 0),
#             "text": (255, 255, 255)
#         }
        
#         self.font = pygame.font.SysFont("Arial", 14)
#         self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        
#         self.city_positions = {}
        
#     def load_game_log(self, log_file):
#         """ゲームログをロード"""
#         with open(log_file, 'r') as f:
#             self.game_data = json.load(f)

#         self.calculate_city_positions()
        
#     def calculate_city_positions(self):
#         """distribute city positions in a circular layout"""
#         cities = self.game_data["cities"]
#         num_cities = len(cities)
        
#         center_x = self.width // 2
#         center_y = self.height // 2
#         radius = min(self.width, self.height) * 0.4
        
#         for i, city in enumerate(cities):
#             angle = 2 * math.pi * i / num_cities
#             x = center_x + radius * math.cos(angle)
#             y = center_y + radius * math.sin(angle)
            
#             self.city_positions[city["name"]] = (int(x), int(y))
    
#     def draw_network(self):
#         for city in self.game_data["cities"]:
#             city_pos = self.city_positions[city["name"]]
            
#             for neighbor in city["neighbors"]:
#                 if neighbor in self.city_positions:
#                     neighbor_pos = self.city_positions[neighbor]
#                     pygame.draw.line(self.screen, (100, 100, 100), city_pos, neighbor_pos, 2)
        
#         for city in self.game_data["cities"]:
#             pos = self.city_positions[city["name"]]
#             infection = city["infection_level"]
            
#             # coloring based on infection level
#             if infection == 0:
#                 color = self.colors["city_normal"]
#             elif infection == 1:
#                 color = self.colors["infection_low"]
#             elif infection == 2:
#                 color = self.colors["infection_med"]
#             else:
#                 color = self.colors["infection_high"]
            
#             pygame.draw.circle(self.screen, color, pos, 15)
            
#             # if research station, draw a square
#             if city["research_station"]:
#                 pygame.draw.rect(self.screen, self.colors["research"], 
#                                (pos[0]-20, pos[1]-20, 40, 40), 2)
            
#             text = self.font.render(city["name"], True, self.colors["text"])
#             self.screen.blit(text, (pos[0] - text.get_width() // 2, pos[1] + 20))
    
#     def draw_players(self):
#         for i, player in enumerate(self.game_data["players"]):
#             if player["city"]:
#                 pos = self.city_positions[player["city"]]
#                 offset_x = (i - len(self.game_data["players"]) / 2) * 10
#                 player_pos = (pos[0] + offset_x, pos[1] - 5)
                
#                 pygame.draw.polygon(self.screen, self.colors["player"],
#                                   [(player_pos[0], player_pos[1] - 15),
#                                    (player_pos[0] - 10, player_pos[1] + 5),
#                                    (player_pos[0] + 10, player_pos[1] + 5)])
                
#                 text = self.font.render(player["name"], True, self.colors["text"])
#                 self.screen.blit(text, (player_pos[0] - text.get_width() // 2, player_pos[1] - 30))
    
#     def draw_status(self):
#         title_text = self.title_font.render(
#             f"Simulation - Turn {self.game_data['turn']}", 
#             True, self.colors["text"])
#         self.screen.blit(title_text, (20, 20))
        
#         outbreak_text = self.font.render(
#             f"Outbreaks: {self.game_data['outbreak_count']}", 
#             True, self.colors["text"])
#         self.screen.blit(outbreak_text, (20, 60))

#         if self.game_data["game_over"]:
#             result = "Wins！" if self.game_data["win"] else "Losses..."
#             result_text = self.title_font.render(result, True, 
#                                             (100, 255, 100) if self.game_data["win"] else (255, 100, 100))
#             self.screen.blit(result_text, (self.width // 2 - result_text.get_width() // 2, 100))
    
#     def run(self, log_file):
#         self.load_game_log(log_file)
        
#         running = True
#         while running:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
            
#             self.screen.fill(self.colors["background"])
            
#             self.draw_network()
#             self.draw_players()
#             self.draw_status()
            
#             pygame.display.flip()
            
#         pygame.quit()

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         log_file = sys.argv[1]
#     else:
#         log_file = "logs/simulation_log.json"
    
#     viewer = PandemicViewer()
#     viewer.run(log_file)