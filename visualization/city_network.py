import networkx as nx
import matplotlib.pyplot as plt
import json

def visualize_game_state(game_log, output_file="city_network.png"):
    """都市ネットワークと感染状況を可視化"""
    # グラフ作成
    G = nx.Graph()
    
    # 都市をノードとして追加
    cities = game_log["cities"]
    for city in cities:
        # 感染レベルによってノードの色を変える
        infection_level = city["infection_level"]
        color = "green" if infection_level == 0 else f"#{min(255, infection_level * 50):02x}0000"
        
        G.add_node(city["name"], 
                  infection=infection_level, 
                  research=city["research_station"],
                  color=color)
    
    # 都市間の接続を追加
    for city in cities:
        for neighbor in city["neighbors"]:
            G.add_edge(city["name"], neighbor)
    
    plt.figure(figsize=(12, 10))
    
    # ノードの色リスト
    node_colors = [G.nodes[node]["color"] for node in G.nodes]
    
    # 研究所のある都市を四角で表示
    node_shapes = []
    for node in G.nodes:
        if G.nodes[node]["research"]:
            node_shapes.append("s")  # 四角
        else:
            node_shapes.append("o")  # 円
    
    # 地域ごとにノードを位置決め
    pos = nx.spring_layout(G, seed=42)
    
    # プレイヤーの位置をマーク
    for player in game_log["players"]:
        if player["city"]:
            plt.plot(pos[player["city"]][0], pos[player["city"]][1], 
                     marker="*", markersize=15, color="blue")
    
    # ネットワーク描画
    nx.draw(G, pos, node_color=node_colors, with_labels=True, 
            font_weight='bold', node_size=700, font_size=8)
    
    # 凡例
    plt.legend(["プレイヤー位置"], loc="upper left")
    
    # タイトル
    plt.title(f"Simulation - Turn {game_log['turn']}")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"ネットワーク図を保存しました: {output_file}")
    plt.close()