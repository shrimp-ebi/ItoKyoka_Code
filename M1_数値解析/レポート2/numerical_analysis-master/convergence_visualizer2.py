# -*- coding: utf-8 -*-
"""
画像比較実験の収束過程可視化プログラム
異なる画像での収束軌跡をグラフで比較表示
"""
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

def extract_image_name_from_folder(folder_name):
    """
    フォルダ名から画像名を抽出
    例: 'Fuji_true_s1.5_t30.0_init_s1.0_t0.0' -> 'Fuji'
    """
    parts = folder_name.split('_')
    if len(parts) > 0:
        return parts[0]
    return None

def load_image_experiment_data(base_dir="output/実験2画像の違い比較"):
    """
    画像比較実験データを読み込み
    """
    experiments = []
    
    if not os.path.exists(base_dir):
        print(f"ディレクトリが見つかりません: {base_dir}")
        return experiments
    
    # 画像比較実験のフォルダを検索
    folders = glob.glob(os.path.join(base_dir, "*_true_s1.5_t30.0_init_s1.0_t0.0"))
    print(f"検索パス: {os.path.join(base_dir, '*_true_s1.5_t30.0_init_s1.0_t0.0')}")
    print(f"見つかったフォルダ数: {len(folders)}")
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        print(f"処理中: {folder_name}")
        
        image_name = extract_image_name_from_folder(folder_name)
        if image_name is None:
            print(f"  画像名抽出失敗")
            continue
        
        # CSVファイルのパス
        result_path = os.path.join(folder, "result_summary.csv")
        history_path = os.path.join(folder, "history.csv")
        
        print(f"  result_path存在確認: {os.path.exists(result_path)}")
        print(f"  history_path存在確認: {os.path.exists(history_path)}")
        
        if not os.path.exists(result_path) or not os.path.exists(history_path):
            print(f"  CSVファイルが見つかりません")
            continue
        
        try:
            # 結果サマリー読み込み
            result_df = pd.read_csv(result_path, encoding='utf-8-sig')
            result = result_df.iloc[0].to_dict()
            
            # 履歴読み込み
            history_df = pd.read_csv(history_path, encoding='utf-8-sig')
            
            experiment = {
                'image_name': image_name,
                'result': result,
                'history': history_df,
                'folder_name': folder_name,
                'folder_path': folder
            }
            
            experiments.append(experiment)
            print(f"  読み込み成功: {image_name}")
            
        except Exception as e:
            print(f"  エラー {folder}: {e}")
    
    return experiments

def get_image_size_info(image_name):
    """
    画像名からサイズ情報を取得
    """
    size_info = {
        'Fuji': '150×150',
        'Fuji2': '1038×1038', 
        'Eevee': '150×150'
    }
    return size_info.get(image_name, '不明')

def get_image_type_info(image_name):
    """
    画像名から特徴情報を取得
    """
    type_info = {
        'Fuji': '自然画像（低解像度）',
        'Fuji2': '自然画像（高解像度）',
        'Eevee': 'キャラクター画像'
    }
    return type_info.get(image_name, '不明')

def plot_image_comparison_convergence(experiments, output_dir="image_comparison_plots"):
    """
    画像比較実験の収束過程プロット
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(experiments) == 0:
        print("プロット対象の実験データがありません")
        return
    
    # 画像名でソート
    experiments.sort(key=lambda x: x['image_name'])
    
    # 1. 角度収束過程の比較
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for i, exp in enumerate(experiments):
        history = exp['history']
        iterations = range(len(history))
        theta_values = history['回転角度θ（deg）']
        
        converged = int(exp['result']['反復回数']) < 999
        style = '-' if converged else '--'
        alpha = 1.0 if converged else 0.7
        linewidth = 2.5 if converged else 2
        
        size_info = get_image_size_info(exp['image_name'])
        label = f"{exp['image_name']} ({size_info})"
        color = colors[i % len(colors)]
        
        plt.plot(iterations, theta_values, style, 
                label=label, linewidth=linewidth, alpha=alpha, color=color)
    
    plt.axhline(y=30.0, color='red', linestyle=':', linewidth=3, label='真値 (30°)')
    plt.xlabel('反復回数')
    plt.ylabel('推定角度θ (度)')
    plt.title('画像比較実験：角度収束過程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_theta_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. スケール収束過程の比較
    plt.figure(figsize=(12, 8))
    
    for i, exp in enumerate(experiments):
        history = exp['history']
        iterations = range(len(history))
        scale_values = history['スケールs']
        
        converged = int(exp['result']['反復回数']) < 999
        style = '-' if converged else '--'
        alpha = 1.0 if converged else 0.7
        linewidth = 2.5 if converged else 2
        
        size_info = get_image_size_info(exp['image_name'])
        label = f"{exp['image_name']} ({size_info})"
        color = colors[i % len(colors)]
        
        plt.plot(iterations, scale_values, style, 
                label=label, linewidth=linewidth, alpha=alpha, color=color)
    
    plt.axhline(y=1.5, color='red', linestyle=':', linewidth=3, label='真値 (1.5)')
    plt.xlabel('反復回数')
    plt.ylabel('推定スケールs')
    plt.title('画像比較実験：スケール収束過程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_scale_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 詳細比較（2x2レイアウト）
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 3-1. 角度収束過程（詳細）
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        history = exp['history']
        iterations = range(len(history))
        theta_values = history['回転角度θ（deg）']
        
        converged = int(exp['result']['反復回数']) < 999
        style = '-' if converged else '--'
        alpha = 1.0 if converged else 0.7
        color = colors[i % len(colors)]
        
        ax1.plot(iterations, theta_values, style, 
                label=f"{exp['image_name']}", linewidth=2, alpha=alpha, color=color)
    
    ax1.axhline(y=30.0, color='red', linestyle=':', linewidth=2, label='真値')
    ax1.set_xlabel('反復回数')
    ax1.set_ylabel('推定角度θ (度)')
    ax1.set_title('角度収束過程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3-2. スケール収束過程（詳細）
    ax2 = axes[0, 1]
    for i, exp in enumerate(experiments):
        history = exp['history']
        iterations = range(len(history))
        scale_values = history['スケールs']
        
        converged = int(exp['result']['反復回数']) < 999
        style = '-' if converged else '--'
        alpha = 1.0 if converged else 0.7
        color = colors[i % len(colors)]
        
        ax2.plot(iterations, scale_values, style, 
                label=f"{exp['image_name']}", linewidth=2, alpha=alpha, color=color)
    
    ax2.axhline(y=1.5, color='red', linestyle=':', linewidth=2, label='真値')
    ax2.set_xlabel('反復回数')
    ax2.set_ylabel('推定スケールs')
    ax2.set_title('スケール収束過程')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3-3. 収束性比較
    ax3 = axes[1, 0]
    image_names = [exp['image_name'] for exp in experiments]
    iterations = [int(exp['result']['反復回数']) for exp in experiments]
    converged_mask = [it < 999 for it in iterations]
    
    bar_colors = ['blue' if c else 'red' for c in converged_mask]
    bars = ax3.bar(image_names, iterations, color=bar_colors, alpha=0.7)
    
    # 反復回数をバーの上に表示
    for i, (bar, iter_count) in enumerate(zip(bars, iterations)):
        height = bar.get_height()
        status = "収束" if converged_mask[i] else "非収束"
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{iter_count}\n({status})', ha='center', va='bottom', fontsize=10)
    
    ax3.set_xlabel('画像')
    ax3.set_ylabel('反復回数')
    ax3.set_title('画像別収束性比較')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1050)
    
    # 3-4. 推定精度比較
    ax4 = axes[1, 1]
    angle_errors = [exp['result']['角度誤差'] for exp in experiments]
    scale_errors = [exp['result']['スケール誤差'] for exp in experiments]
    
    for i, exp in enumerate(experiments):
        color = colors[i % len(colors)]
        marker = 'o' if converged_mask[i] else 'x'
        size = 150 if converged_mask[i] else 100
        
        ax4.scatter(angle_errors[i], scale_errors[i], 
                   c=color, s=size, alpha=0.8, marker=marker, 
                   label=f"{exp['image_name']}")
        
        # 画像名をアノテーション
        ax4.annotate(exp['image_name'], 
                    (angle_errors[i], scale_errors[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('角度推定誤差 (度)')
    ax4.set_ylabel('スケール推定誤差')
    ax4.set_title('推定精度比較')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_detailed_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"画像比較プロットを保存: {output_dir}/image_theta_convergence.png")
    print(f"画像比較プロットを保存: {output_dir}/image_scale_convergence.png") 
    print(f"画像比較プロットを保存: {output_dir}/image_detailed_comparison.png")

def create_image_comparison_table(experiments, output_dir="image_comparison_plots"):
    """
    画像比較実験の結果テーブルを作成
    """
    os.makedirs(output_dir, exist_ok=True)
    
    experiments.sort(key=lambda x: x['image_name'])
    
    print("\n=== 画像比較実験結果 ===")
    print("画像名 | サイズ | 画像特性 | 推定角度[度] | 推定スケール | 角度誤差[度] | スケール誤差 | 反復回数 | 収束判定")
    print("-" * 120)
    
    latex_table = []
    for exp in experiments:
        iterations = int(exp['result']['反復回数'])
        converged = "収束" if iterations < 999 else "非収束"
        size_info = get_image_size_info(exp['image_name'])
        type_info = get_image_type_info(exp['image_name'])
        
        print(f"{exp['image_name']:8s} | {size_info:9s} | {type_info:15s} | "
              f"{exp['result']['推定 角度(deg)']:10.6f} | {exp['result']['推定 スケール']:11.6f} | "
              f"{exp['result']['角度誤差']:11.6f} | {exp['result']['スケール誤差']:12.6f} | "
              f"{iterations:6d} | {converged}")
        
        # LaTeX用データ
        latex_table.append([
            exp['image_name'],
            size_info,
            exp['result']['推定 角度(deg)'],
            exp['result']['推定 スケール'],
            exp['result']['角度誤差'],
            exp['result']['スケール誤差'],
            iterations
        ])
    
    # LaTeX テーブル生成
    with open(os.path.join(output_dir, 'image_comparison_table.tex'), 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{入力画像による推定精度の比較}\n")
        f.write("\\label{tab:image_comparison}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("画像名 & 画像サイズ & 推定角度[度] & 推定スケール & 角度誤差[度] & スケール誤差 & 反復回数 \\\\\n")
        f.write("\\hline\n")
        
        for row in latex_table:
            converged_mark = "" if row[6] < 999 else "**"
            f.write(f"{row[0]} & {row[1]} & {row[2]:.6f} & {row[3]:.6f} & {row[4]:.6f} & {row[5]:.6f} & {row[6]}{converged_mark} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("\n**最大反復回数に達し収束せず\n")
    
    print(f"\nLaTeXテーブルを保存: {output_dir}/image_comparison_table.tex")

def analyze_image_characteristics(experiments):
    """
    画像特性と収束性の関係を分析
    """
    print("\n=== 画像特性分析 ===")
    
    converged_count = 0
    total_count = len(experiments)
    
    for exp in experiments:
        iterations = int(exp['result']['反復回数'])
        converged = iterations < 999
        if converged:
            converged_count += 1
        
        type_info = get_image_type_info(exp['image_name'])
        size_info = get_image_size_info(exp['image_name'])
        
        print(f"\n【{exp['image_name']}画像】")
        print(f"- 特性: {type_info}")
        print(f"- サイズ: {size_info}")
        print(f"- 収束判定: {'成功' if converged else '失敗'} ({iterations}回)")
        print(f"- 最終推定精度: 角度誤差{exp['result']['角度誤差']:.6f}度, スケール誤差{exp['result']['スケール誤差']:.6f}")
    
    print(f"\n【総合評価】")
    print(f"収束成功率: {converged_count}/{total_count} ({converged_count/total_count*100:.1f}%)")
    
    # 画像特性と成功率の関係
    success_by_type = {}
    for exp in experiments:
        type_info = get_image_type_info(exp['image_name'])
        iterations = int(exp['result']['反復回数'])
        converged = iterations < 999
        
        if type_info not in success_by_type:
            success_by_type[type_info] = {'success': 0, 'total': 0}
        
        success_by_type[type_info]['total'] += 1
        if converged:
            success_by_type[type_info]['success'] += 1
    
    print(f"\n【画像タイプ別成功率】")
    for img_type, stats in success_by_type.items():
        rate = stats['success'] / stats['total'] * 100
        print(f"- {img_type}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

def main():
    """
    メイン処理
    """
    print("=== 画像比較実験の収束過程可視化開始 ===")
    
    # データ読み込み
    experiments = load_image_experiment_data()
    
    if len(experiments) == 0:
        print("実験データが見つかりません")
        return
    
    print(f"\n読み込み完了: {len(experiments)}件の実験データ")
    
    # 収束過程のプロット
    plot_image_comparison_convergence(experiments)
    
    # 結果テーブル作成
    create_image_comparison_table(experiments)
    
    # 画像特性分析
    analyze_image_characteristics(experiments)
    
    print("\n=== 可視化完了 ===")
    print("以下のファイルが生成されました:")
    print("- image_comparison_plots/image_theta_convergence.png")
    print("- image_comparison_plots/image_scale_convergence.png")
    print("- image_comparison_plots/image_detailed_comparison.png")
    print("- image_comparison_plots/image_comparison_table.tex")

if __name__ == "__main__":
    main()