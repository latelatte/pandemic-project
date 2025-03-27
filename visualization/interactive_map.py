import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

def visualize_interactive_map(game_log_file, output_file="pandemic_map.html"):
    """display the game state on an interactive map"""
    with open(game_log_file, 'r') as f:
        game_log = json.load(f)
    
    cities_data = []
    for city in game_log["cities"]:
        # TODO: use precise coordinates of cities
        city_coords = {
            "Atlanta": (33.7490, -84.3880),
            "Chicago": (41.8781, -87.6298),
            "New York": (40.7128, -74.0060),
            "Washington": (38.9072, -77.0369),
            "San Francisco": (37.7749, -122.4194),
            "London": (51.5074, -0.1278),
            "Paris": (48.8566, 2.3522),
            "Madrid": (40.4168, -3.7038),
            "Tokyo": (35.6762, 139.6503),
            "Sydney": (-33.8688, 151.2093),
            # TODO: add more cities
        }.get(city["name"], (0, 0)) # default to (0, 0) if city not found

        cities_data.append({
            "name": city["name"],
            "lat": city_coords[0],
            "lon": city_coords[1],
            "infection": city["infection_level"],
            "research_station": city["research_station"]
        })
    
    df = pd.DataFrame(cities_data)
    
    # config for color and size
    df["size"] = df["infection"] * 10 + 15
    df["color"] = df["infection"].apply(lambda x: f"rgb({min(255, x*80)}, 0, 0)")
    
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        hover_name="name",
        size="size",
        color="infection",
        color_continuous_scale="Reds",
        projection="natural earth",
        title=f"Simulation State（Turn {game_log['turn']}）"
    )
    
    # marker for reserach stations
    research_stations = df[df["research_station"] == True]
    fig.add_trace(
        go.Scattergeo(
            lat=research_stations["lat"],
            lon=research_stations["lon"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=12,
                color="blue"
            ),
            name="reserach station",
        )
    )
    
    # playwe location
    player_positions = []
    for player in game_log["players"]:
        if player["city"]:
            city_info = next((c for c in cities_data if c["name"] == player["city"]), None)
            if city_info:
                player_positions.append({
                    "name": player["name"],
                    "lat": city_info["lat"],
                    "lon": city_info["lon"]
                })
    
    if player_positions:
        player_df = pd.DataFrame(player_positions)
        fig.add_trace(
            go.Scattergeo(
                lat=player_df["lat"],
                lon=player_df["lon"],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=15,
                    color="yellow"
                ),
                name="player"
            )
        )
    
    # 保存
    fig.write_html(output_file)
    print(f"created interactive map: {output_file}")