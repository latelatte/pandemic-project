import argparse
import os
import sys
import json
from visualization.city_network import visualize_game_state
from visualization.interactive_map import visualize_interactive_map
from visualization.performance_charts import create_performance_charts
from visualization.game_viewer import PandemicViewer

def main():
    parser = argparse.ArgumentParser(description="Pandemic Simulation Visualization Tools")
    parser.add_argument("--log-file", help="Path to the game log JSON file")
    parser.add_argument("--results-dir", help="Directory containing results for multiple algorithms")
    parser.add_argument("--type", choices=["network", "map", "charts", "game"], default="network",
                        help="Type of visualization to generate")
    parser.add_argument("--output", help="Output file/directory")
    
    args = parser.parse_args()
    
    # 出力先の設定
    output = args.output or "./visualization_output"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    if args.type == "network":
        if not args.log_file:
            print("Error: --log-file required for network visualization")
            sys.exit(1)
            
        with open(args.log_file) as f:
            game_log = json.load(f)
        
        visualize_game_state(game_log, output or "city_network.png")
        
    elif args.type == "map":
        if not args.log_file:
            print("Error: --log-file required for map visualization")
            sys.exit(1)
            
        visualize_interactive_map(args.log_file, output or "pandemic_map.html")
        
    elif args.type == "charts":
        if not args.results_dir:
            print("Error: --results-dir required for performance charts")
            sys.exit(1)
            
        create_performance_charts(args.results_dir, output or "./plots")
        
    elif args.type == "game":
        if not args.log_file:
            print("Error: --log-file required for game viewer")
            sys.exit(1)
            
        viewer = PandemicViewer()
        viewer.run(args.log_file)

if __name__ == "__main__":
    main()