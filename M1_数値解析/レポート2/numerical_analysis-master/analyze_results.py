# -*- coding: utf-8 -*-
"""
実験結果自動収集・分析プログラム
gauss.pyで生成された結果を自動的に収集し、レポート用のテーブルを作成する
"""
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

def collect_experiment_results(output_dir="output"):
    """
    outputディレクトリから全ての実験結果を収集する
    """
    results = []
    
    # 全てのresult_summary.csvファイルを検索
    csv_files = glob.glob(os.path.join(output_dir, "**/result_summary.csv"), recursive=True)
    
    for csv_file in csv_files:
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            if len(df) > 0:
                result = df.iloc[0].to_dict()
                
                # フォルダ名から実験情報を抽出
                folder_name = os.path.basename(os.path.dirname(csv_file))
                result['folder_name'] = folder_name
                
                # 実験タイプを判定
                result['experiment_type'] = classify_experiment_type(result)
                
                results.append(result)
                print(f"収集: {folder_name}")
                
        except Exception as e:
            print(f"エラー: {csv_file} - {e}")
    
    return pd.DataFrame(results)

def classify_experiment_type(result):
    """
    実験結果から実験タイプを分類する
    """
    theta_init = result['初期値 角度']
    scale_init = result['初期値 スケール']
    
    # 基本実験
    if theta_init == 0.0 and scale_init == 1.0:
        return "基本実験"
    
    # スケール初期値実験（角度が真値30度固定）
    elif theta_init == 30.0:
        return "スケール初期値実験"
    
    # 角度初期値実験（スケールが真値1.5固定）
    elif scale_init == 1.5:
        return "角度初期値実験"
    
    # その他
    else:
        return "その他実験"

def create_scale_init_table(df):
    """
    スケール初期値実験の結果テーブルを作成
    """
    scale_exp = df[df['experiment_type'] == 'スケール初期値実験'].copy()
    scale_exp = scale_exp.sort_values('初期値 スケール')
    
    print("\n=== スケール初期値実験結果 ===")
    print("初期スケール | 推定角度[度] | 推定スケール | 角度誤差[度] | スケール誤差 | 反復回数")
    print("-" * 80)
    
    table_data = []
    for _, row in scale_exp.iterrows():
        print(f"{row['初期値 スケール']:8.1f} | {row['推定 角度(deg)']:11.6f} | {row['推定 スケール']:11.6f} | {row['角度誤差']:11.6f} | {row['スケール誤差']:10.6f} | {int(row['反復回数']):6d}")
        table_data.append([
            row['初期値 スケール'],
            row['推定 角度(deg)'],
            row['推定 スケール'],
            row['角度誤差'],
            row['スケール誤差'],
            int(row['反復回数'])
        ])
    
    return table_data

def create_theta_init_table(df):
    """
    角度初期値実験の結果テーブルを作成
    """
    theta_exp = df[df['experiment_type'] == '角度初期値実験'].copy()
    theta_exp = theta_exp.sort_values('初期値 角度')
    
    print("\n=== 角度初期値実験結果 ===")
    print("初期角度[度] | 推定角度[度] | 推定スケール | 角度誤差[度] | スケール誤差 | 反復回数")
    print("-" * 80)
    
    table_data = []
    for _, row in theta_exp.iterrows():
        print(f"{row['初期値 角度']:10.1f} | {row['推定 角度(deg)']:11.6f} | {row['推定 スケール']:11.6f} | {row['角度誤差']:11.6f} | {row['スケール誤差']:10.6f} | {int(row['反復回数']):6d}")
        table_data.append([
            row['初期値 角度'],
            row['推定 角度(deg)'],
            row['推定 スケール'],
            row['角度誤差'],
            row['スケール誤差'],
            int(row['反復回数'])
        ])
    
    return table_data

def create_convergence_comparison_plot(df, output_path="analysis_results"):
    """
    収束過程の比較プロットを作成
    """
    os.makedirs(output_path, exist_ok=True)
    
    # スケール初期値実験の収束比較
    scale_exp = df[df['experiment_type'] == 'スケール初期値実験'].copy()
    if len(scale_exp) > 0:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for _, row in scale_exp.iterrows():
            plt.scatter(row['初期値 スケール'], row['反復回数'], 
                       s=100, alpha=0.7, label=f"s_init={row['初期値 スケール']}")
        plt.xlabel('初期スケール値')
        plt.ylabel('収束反復回数')
        plt.title('スケール初期値 vs 収束速度')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        for _, row in scale_exp.iterrows():
            plt.scatter(row['初期値 スケール'], row['角度誤差'], 
                       s=100, alpha=0.7)
        plt.xlabel('初期スケール値')
        plt.ylabel('角度推定誤差[度]')
        plt.title('スケール初期値 vs 角度推定精度')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'scale_init_analysis.png'), dpi=300)
        plt.close()
    
    # 角度初期値実験の収束比較
    theta_exp = df[df['experiment_type'] == '角度初期値実験'].copy()
    if len(theta_exp) > 0:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for _, row in theta_exp.iterrows():
            plt.scatter(row['初期値 角度'], row['反復回数'], 
                       s=100, alpha=0.7)
        plt.xlabel('初期角度[度]')
        plt.ylabel('収束反復回数')
        plt.title('角度初期値 vs 収束速度')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        for _, row in theta_exp.iterrows():
            plt.scatter(row['初期値 角度'], row['スケール誤差'], 
                       s=100, alpha=0.7)
        plt.xlabel('初期角度[度]')
        plt.ylabel('スケール推定誤差')
        plt.title('角度初期値 vs スケール推定精度')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'theta_init_analysis.png'), dpi=300)
        plt.close()

def generate_latex_tables(df, output_path="analysis_results"):
    """
    LaTeX用のテーブルコードを生成
    """
    os.makedirs(output_path, exist_ok=True)
    
    # スケール初期値実験テーブル
    scale_exp = df[df['experiment_type'] == 'スケール初期値実験'].copy()
    scale_exp = scale_exp.sort_values('初期値 スケール')
    
    with open(os.path.join(output_path, 'scale_init_table.tex'), 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{スケール初期値による収束特性（Fuji.jpg画像）}\n")
        f.write("\\label{tab:scale_init_results}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("初期スケール & 推定角度[度] & 推定スケール & 角度誤差[度] & スケール誤差 & 反復回数 \\\\\n")
        f.write("\\hline\n")
        
        for _, row in scale_exp.iterrows():
            f.write(f"{row['初期値 スケール']:.1f} & {row['推定 角度(deg)']:.6f} & {row['推定 スケール']:.6f} & {row['角度誤差']:.6f} & {row['スケール誤差']:.6f} & {int(row['反復回数'])} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # 角度初期値実験テーブル
    theta_exp = df[df['experiment_type'] == '角度初期値実験'].copy()
    theta_exp = theta_exp.sort_values('初期値 角度')
    
    with open(os.path.join(output_path, 'theta_init_table.tex'), 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{角度初期値による収束特性（Fuji.jpg画像）}\n")
        f.write("\\label{tab:theta_init_results}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("初期角度[度] & 推定角度[度] & 推定スケール & 角度誤差[度] & スケール誤差 & 反復回数 \\\\\n")
        f.write("\\hline\n")
        
        for _, row in theta_exp.iterrows():
            f.write(f"{row['初期値 角度']:.0f} & {row['推定 角度(deg)']:.6f} & {row['推定 スケール']:.6f} & {row['角度誤差']:.6f} & {row['スケール誤差']:.6f} & {int(row['反復回数'])} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def analyze_convergence_characteristics(df):
    """
    収束特性の統計分析
    """
    print("\n=== 収束特性統計分析 ===")
    
    # スケール初期値実験の分析
    scale_exp = df[df['experiment_type'] == 'スケール初期値実験']
    if len(scale_exp) > 0:
        print("\n【スケール初期値実験】")
        print(f"平均反復回数: {scale_exp['反復回数'].mean():.1f}")
        print(f"最小反復回数: {scale_exp['反復回数'].min()}")
        print(f"最大反復回数: {scale_exp['反復回数'].max()}")
        print(f"平均角度誤差: {scale_exp['角度誤差'].mean():.6f}度")
        print(f"平均スケール誤差: {scale_exp['スケール誤差'].mean():.6f}")
        
        # 最適初期値の特定
        best_scale = scale_exp.loc[scale_exp['反復回数'].idxmin()]
        print(f"最速収束の初期スケール: {best_scale['初期値 スケール']} (反復回数: {best_scale['反復回数']})")
    
    # 角度初期値実験の分析
    theta_exp = df[df['experiment_type'] == '角度初期値実験']
    if len(theta_exp) > 0:
        print("\n【角度初期値実験】")
        print(f"平均反復回数: {theta_exp['反復回数'].mean():.1f}")
        print(f"最小反復回数: {theta_exp['反復回数'].min()}")
        print(f"最大反復回数: {theta_exp['反復回数'].max()}")
        print(f"平均角度誤差: {theta_exp['角度誤差'].mean():.6f}度")
        print(f"平均スケール誤差: {theta_exp['スケール誤差'].mean():.6f}")
        
        # 最適初期値の特定
        best_theta = theta_exp.loc[theta_exp['反復回数'].idxmin()]
        print(f"最速収束の初期角度: {best_theta['初期値 角度']}度 (反復回数: {best_theta['反復回数']})")

def main():
    """
    メイン処理
    """
    print("=== 実験結果自動収集・分析開始 ===")
    
    # 結果収集
    df = collect_experiment_results()
    
    if len(df) == 0:
        print("実験結果が見つかりません。outputディレクトリを確認してください。")
        return
    
    print(f"\n総実験数: {len(df)}")
    print(f"実験タイプ別内訳:")
    for exp_type in df['experiment_type'].unique():
        count = len(df[df['experiment_type'] == exp_type])
        print(f"  {exp_type}: {count}件")
    
    # テーブル作成
    scale_table = create_scale_init_table(df)
    theta_table = create_theta_init_table(df)
    
    # グラフ作成
    create_convergence_comparison_plot(df)
    
    # LaTeXテーブル生成
    generate_latex_tables(df)
    
    # 統計分析
    analyze_convergence_characteristics(df)
    
    # 結果をCSVで保存
    df.to_csv('analysis_results/all_experiment_results.csv', index=False, encoding='utf-8-sig')
    
    print("\n=== 分析完了 ===")
    print("結果は以下に保存されました:")
    print("- analysis_results/scale_init_table.tex")
    print("- analysis_results/theta_init_table.tex") 
    print("- analysis_results/scale_init_analysis.png")
    print("- analysis_results/theta_init_analysis.png")
    print("- analysis_results/all_experiment_results.csv")

if __name__ == "__main__":
    main()