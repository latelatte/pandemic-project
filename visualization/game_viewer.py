import pygame
import sys
import json
import math

class PandemicViewer:
    """PyGameを使ったパンデミックシミュレーションビューワー"""
    def __init__(self, width=1024, height=768):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pandemic Simulation Viewer")
        
        self.width = width
        self.height = height
        
        # 色の定義
        self.colors = {
            "background": (30, 30, 50),
            "city_normal": (200, 200, 200),
            "infection_low": (220, 150, 150),
            "infection_med": (240, 100, 100),
            "infection_high": (255, 50, 50),
            "research": (100, 150, 255),
            "player": (255, 255, 0),
            "text": (255, 255, 255)
        }
        
        # フォント初期化
        self.font = pygame.font.SysFont("Arial", 14)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        
        # 都市の座標マップ（仮値、実際は地図上の適切な位置に配置）
        self.city_positions = {}
        
    def load_game_log(self, log_file):
        """ゲームログをロード"""
        with open(log_file, 'r') as f:
            self.game_data = json.load(f)
            
        # 都市位置を自動計算（円環状に配置）
        self.calculate_city_positions()
        
    def calculate_city_positions(self):
        """都市を円環状に自動配置"""
        cities = self.game_data["cities"]
        num_cities = len(cities)
        
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(self.width, self.height) * 0.4
        
        for i, city in enumerate(cities):
            angle = 2 * math.pi * i / num_cities
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            self.city_positions[city["name"]] = (int(x), int(y))
    
    def draw_network(self):
        """都市ネットワークを描画"""
        # 接続線を描画
        for city in self.game_data["cities"]:
            city_pos = self.city_positions[city["name"]]
            
            for neighbor in city["neighbors"]:
                if neighbor in self.city_positions:
                    neighbor_pos = self.city_positions[neighbor]
                    pygame.draw.line(self.screen, (100, 100, 100), city_pos, neighbor_pos, 2)
        
        # 都市を描画
        for city in self.game_data["cities"]:
            pos = self.city_positions[city["name"]]
            infection = city["infection_level"]
            
            # 感染レベルに応じて色を決定
            if infection == 0:
                color = self.colors["city_normal"]
            elif infection == 1:
                color = self.colors["infection_low"]
            elif infection == 2:
                color = self.colors["infection_med"]
            else:
                color = self.colors["infection_high"]
            
            # 都市の円を描画
            pygame.draw.circle(self.screen, color, pos, 15)
            
            # 研究所がある場合は四角で囲む
            if city["research_station"]:
                pygame.draw.rect(self.screen, self.colors["research"], 
                               (pos[0]-20, pos[1]-20, 40, 40), 2)
            
            # 都市名を描画
            text = self.font.render(city["name"], True, self.colors["text"])
            self.screen.blit(text, (pos[0] - text.get_width() // 2, pos[1] + 20))
    
    def draw_players(self):
        """プレイヤーを描画"""
        for i, player in enumerate(self.game_data["players"]):
            if player["city"]:
                pos = self.city_positions[player["city"]]
                # プレイヤーごとに少し位置をずらす
                offset_x = (i - len(self.game_data["players"]) / 2) * 10
                player_pos = (pos[0] + offset_x, pos[1] - 5)
                
                # プレイヤーマーカー描画
                pygame.draw.polygon(self.screen, self.colors["player"],
                                  [(player_pos[0], player_pos[1] - 15),
                                   (player_pos[0] - 10, player_pos[1] + 5),
                                   (player_pos[0] + 10, player_pos[1] + 5)])
                
                # プレイヤー名描画
                text = self.font.render(player["name"], True, self.colors["text"])
                self.screen.blit(text, (player_pos[0] - text.get_width() // 2, player_pos[1] - 30))
    
    def draw_status(self):
        """ゲーム状態を描画"""
        # タイトル
        title_text = self.title_font.render(
            f"Simulation - Turn {self.game_data['turn']}", 
            True, self.colors["text"])
        self.screen.blit(title_text, (20, 20))
        
        # アウトブレイク数
        outbreak_text = self.font.render(
            f"Outbreaks: {self.game_data['outbreak_count']}", 
            True, self.colors["text"])
        self.screen.blit(outbreak_text, (20, 60))
        
        # ゲーム終了状態
        if self.game_data["game_over"]:
            result = "Wins！" if self.game_data["win"] else "Losses..."
            result_text = self.title_font.render(result, True, 
                                            (100, 255, 100) if self.game_data["win"] else (255, 100, 100))
            self.screen.blit(result_text, (self.width // 2 - result_text.get_width() // 2, 100))
    
    def run(self, log_file):
        """ビューワーを実行"""
        self.load_game_log(log_file)
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # 画面をクリア
            self.screen.fill(self.colors["background"])
            
            # 要素を描画
            self.draw_network()
            self.draw_players()
            self.draw_status()
            
            pygame.display.flip()
            
        pygame.quit()

# テスト用メイン処理
if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "logs/simulation_log.json"
    
    viewer = PandemicViewer()
    viewer.run(log_file)