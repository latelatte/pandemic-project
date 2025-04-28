# import networkx as nx
# import matplotlib.pyplot as plt
# import json

# def visualize_game_state(game_log, output_file="city_network.png"):
#     """visualize the game state as a city network"""
#     G = nx.Graph()
    
#     cities = game_log["cities"]
#     for city in cities:
#         # change color based on infection level
#         infection_level = city["infection_level"]
#         color = "green" if infection_level == 0 else f"#{min(255, infection_level * 50):02x}0000"
        
#         G.add_node(city["name"], 
#                   infection=infection_level, 
#                   research=city["research_station"],
#                   color=color)
    
#     for city in cities:
#         for neighbor in city["neighbors"]:
#             G.add_edge(city["name"], neighbor)
    
#     plt.figure(figsize=(12, 10))
    
#     node_colors = [G.nodes[node]["color"] for node in G.nodes]
    
#     # distinguish between research stations and normal cities
#     node_shapes = []
#     for node in G.nodes:
#         if G.nodes[node]["research"]:
#             node_shapes.append("s")
#         else:
#             node_shapes.append("o")
    
#     pos = nx.spring_layout(G, seed=42)
    
#     for player in game_log["players"]:
#         if player["city"]:
#             plt.plot(pos[player["city"]][0], pos[player["city"]][1], 
#                      marker="*", markersize=15, color="blue")
    
#     nx.draw(G, pos, node_color=node_colors, with_labels=True, 
#             font_weight='bold', node_size=700, font_size=8)
    
#     plt.legend(["Player location"], loc="upper left")
    
#     plt.title(f"Simulation - Turn {game_log['turn']}")
    
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"network image saved: {output_file}")
#     plt.close()