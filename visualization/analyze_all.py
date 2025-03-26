import os
from visualization.pareto_analysis import run_pareto_analysis
from visualization.cnp_analysis import run_cnp_analysis

def run_all_analyses(results_dir, output_dir=None):
    """すべての分析を一括実行"""
    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # パレート分析
    pareto_success = run_pareto_analysis(results_dir, output_dir)
    
    # コスト効率性分析
    cnp_success = run_cnp_analysis(results_dir, output_dir)
    
    return pareto_success and cnp_success

if __name__ == "__main__":
    import sys
    
    # 最新の実験ディレクトリを探す
    def find_latest_experiment():
        log_dirs = sorted([d for d in os.listdir("./logs") if d.startswith("experiment_")])
        return os.path.join("./logs", log_dirs[-1]) if log_dirs else None
    
    # コマンドライン引数かデフォルトで最新を使用
    results_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_experiment()
    
    if results_dir:
        print(f"分析対象ディレクトリ: {results_dir}")
        success = run_all_analyses(results_dir)
        if success:
            print(f"すべての分析が完了しました: {os.path.join(results_dir, 'plots')}")
        else:
            print("分析中にエラーが発生しました")
    else:
        print("実験ディレクトリが見つかりません")