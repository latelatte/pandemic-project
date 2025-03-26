import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

def visualize_interactive_map(game_log_file, output_file="pandemic_map.html"):
    """世界地図上にインタラクティブな感染状況を表示"""
    # ゲームログ読み込み
    with open(game_log_file, 'r') as f:
        game_log = json.load(f)
    
    # 都市情報の抽出
    cities_data = []
    for city in game_log["cities"]:
        # 実際のゲームではもっと多くの都市がある
        # 適当な座標を割り当て（実際には正確な緯度経度を使うべき）
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
            # その他の都市も追加...
        }.get(city["name"], (0, 0))  # デフォルト座標

        cities_data.append({
            "name": city["name"],
            "lat": city_coords[0],
            "lon": city_coords[1],
            "infection": city["infection_level"],
            "research_station": city["research_station"]
        })
    
    # データフレーム作成
    df = pd.DataFrame(cities_data)
    
    # マーカーサイズと色設定
    df["size"] = df["infection"] * 10 + 15
    df["color"] = df["infection"].apply(lambda x: f"rgb({min(255, x*80)}, 0, 0)")
    
    # 地図作成
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
    
    # 研究所マーカー追加
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
            name="研究施設"
        )
    )
    
    # プレイヤー位置表示
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
                name="プレイヤー"
            )
        )
    
    # 保存
    fig.write_html(output_file)
    print(f"インタラクティブマップを保存しました: {output_file}")