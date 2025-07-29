# -*- coding: utf-8 -*-
"""
実験3・4の可視化・分析スクリプト
"""
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import re

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

def extract_experiment_info(folder_name, parent_folder=""):
    """
    フォルダ名から実験情報を抽出（階層構造に対応）
    """
    # 実験3: カーネルサイズ実験（新しい階層構造）
    if parent_folder.startswith('kernel'):
        kernel_size = int(parent_folder.replace('kernel', ''))
        return 'kernel', kernel_size, None, None, None
    
    # 従来のkernel検出（フォルダ名に含まれる場合）
    if 'kernel' in folder_name.lower():
        kernel_match = re.search(r'kernel_?(\d+)', folder_name)
        if kernel_match:
            return 'kernel', int(kernel_match.group(1)), None, None, None
    
    # 実験4: 極端パラメータ実験
    # true_s2.0_t30.0 のような形式
    true_match = re.search(r'true_s([\d.]+)_t([\d.-]+)', folder_name)
    init_match = re.search(r'init_s([\d.]+)_t([\d.-]+)', folder_name)
    
    if true_match and init_match:
        true_scale = float(true_match.group(1))
        true_angle = float(true_match.group(2))
        init_scale = float(init_match.group(1))
        init_angle = float(init_match.group(2))
        return 'extreme', None, true_scale, true_angle, (init_scale, init_angle)
    
    return None, None, None, None, None

def load_experiment_data(base_dirs):
    """
    複数のディレクトリから実験データを読み込み（階層構造対応）
    """
    experiments = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"ディレクトリが見つかりません: {base_dir}")
            continue
        
        # 第一階層のフォルダ検索
        first_level_folders = glob.glob(os.path.join(base_dir, "*"))
        print(f"{base_dir}: {len(first_level_folders)}個のフォルダを発見")
        
        for first_folder in first_level_folders:
            if not os.path.isdir(first_folder):
                continue
            
            first_folder_name = os.path.basename(first_folder)
            
            # kernel3, kernel5などの場合は第二階層を探索
            if first_folder_name.startswith('kernel'):
                second_level_folders = glob.glob(os.path.join(first_folder, "*"))
                for second_folder in second_level_folders:
                    if not os.path.isdir(second_folder):
                        continue
                    
                    second_folder_name = os.path.basename(second_folder)
                    exp_type, kernel_size, true_scale, true_angle, init_values = extract_experiment_info(
                        second_folder_name, first_folder_name)
                    
                    if exp_type is None:
                        continue
                    
                    # CSVファイルのパス
                    result_path = os.path.join(second_folder, "result_summary.csv")
                    history_path = os.path.join(second_folder, "history.csv")
                    
                    if not os.path.exists(result_path) or not os.path.exists(history_path):
                        print(f"CSVファイルが見つかりません: {first_folder_name}/{second_folder_name}")
                        continue
                    
                    try:
                        # 結果サマリー読み込み
                        result_df = pd.read_csv(result_path, encoding='utf-8-sig')
                        result = result_df.iloc[0].to_dict()
                        
                        # 履歴読み込み
                        history_df = pd.read_csv(history_path, encoding='utf-8-sig')
                        
                        experiment = {
                            'exp_type': exp_type,
                            'kernel_size': kernel_size,
                            'true_scale': true_scale,
                            'true_angle': true_angle,
                            'init_values': init_values,
                            'result': result,
                            'history': history_df,
                            'folder_name': f"{first_folder_name}/{second_folder_name}",
                            'folder_path': second_folder
                        }
                        
                        experiments.append(experiment)
                        print(f"読み込み成功: {first_folder_name}/{second_folder_name}")
                        
                    except Exception as e:
                        print(f"エラー {first_folder_name}/{second_folder_name}: {e}")
            
            else:
                # 直接の第一階層フォルダ（実験4など）
                exp_type, kernel_size, true_scale, true_angle, init_values = extract_experiment_info(first_folder_name)
                
                if exp_type is None:
                    continue
                
                # CSVファイルのパス
                result_path = os.path.join(first_folder, "result_summary.csv")
                history_path = os.path.join(first_folder, "history.csv")
                
                if not os.path.exists(result_path) or not os.path.exists(history_path):
                    print(f"CSVファイルが見つかりません: {first_folder_name}")
                    continue
                
                try:
                    # 結果サマリー読み込み
                    result_df = pd.read_csv(result_path, encoding='utf-8-sig')
                    result = result_df.iloc[0].to_dict()
                    
                    # 履歴読み込み
                    history_df = pd.read_csv(history_path, encoding='utf-8-sig')
                    
                    experiment = {
                        'exp_type': exp_type,
                        'kernel_size': kernel_size,
                        'true_scale': true_scale,
                        'true_angle': true_angle,
                        'init_values': init_values,
                        'result': result,
                        'history': history_df,
                        'folder_name': first_folder_name,
                        'folder_path': first_folder
                    }
                    
                    experiments.append(experiment)
                    print(f"読み込み成功: {first_folder_name}")
                    
                except Exception as e:
                    print(f"エラー {first_folder_name}: {e}")
    
    return experiments

def plot_kernel_size_comparison(experiments, output_dir="visualization_results"):
    """
    実験3: カーネルサイズ比較
    """
    os.makedirs(output_dir, exist_ok=True)
    
    kernel_experiments = [exp for exp in experiments if exp['exp_type'] == 'kernel']
    if len(kernel_experiments) == 0:
        print("カーネルサイズ実験のデータが見つかりません")
        return
    
    kernel_experiments.sort(key=lambda x: x['kernel_size'])
    
    # 1. 収束過程の比較
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 角度収束過程
    ax1 = axes[0]
    for exp in kernel_experiments:
        history = exp['history']
        iterations = range(len(history))
        theta_values = history['回転角度θ（deg）']
        
        label = f"カーネル{exp['kernel_size']}×{exp['kernel_size']}"
        ax1.plot(iterations, theta_values, linewidth=2, label=label)
    
    ax1.axhline(y=30.0, color='red', linestyle=':', linewidth=2, label='真値 (30°)')
    ax1.set_xlabel('反復回数')
    ax1.set_ylabel('推定角度θ (度)')
    ax1.set_title('カーネルサイズ別：角度収束過程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # スケール収束過程
    ax2 = axes[1]
    for exp in kernel_experiments:
        history = exp['history']
        iterations = range(len(history))
        scale_values = history['スケールs']
        
        label = f"カーネル{exp['kernel_size']}×{exp['kernel_size']}"
        ax2.plot(iterations, scale_values, linewidth=2, label=label)
    
    ax2.axhline(y=1.5, color='red', linestyle=':', linewidth=2, label='真値 (1.5)')
    ax2.set_xlabel('反復回数')
    ax2.set_ylabel('推定スケールs')
    ax2.set_title('カーネルサイズ別：スケール収束過程')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kernel_size_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 推定精度と反復回数の比較
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    kernel_sizes = [exp['kernel_size'] for exp in kernel_experiments]
    angle_errors = [exp['result']['角度誤差'] for exp in kernel_experiments]
    scale_errors = [exp['result']['スケール誤差'] for exp in kernel_experiments]
    iterations = [exp['result']['反復回数'] for exp in kernel_experiments]
    
    # 推定精度
    ax1 = axes[0]
    x = np.arange(len(kernel_sizes))
    width = 0.35
    
    ax1.bar(x - width/2, angle_errors, width, label='角度誤差', alpha=0.8)
    ax1.bar(x + width/2, scale_errors, width, label='スケール誤差', alpha=0.8)
    ax1.set_xlabel('カーネルサイズ')
    ax1.set_ylabel('推定誤差')
    ax1.set_title('カーネルサイズ別推定精度')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{k}×{k}' for k in kernel_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 反復回数
    ax2 = axes[1]
    ax2.bar(range(len(kernel_sizes)), iterations, alpha=0.8, color='orange')
    ax2.set_xlabel('カーネルサイズ')
    ax2.set_ylabel('収束反復回数')
    ax2.set_title('カーネルサイズ別収束効率')
    ax2.set_xticks(range(len(kernel_sizes)))
    ax2.set_xticklabels([f'{k}×{k}' for k in kernel_sizes])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kernel_size_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"カーネルサイズ比較グラフを保存: {output_dir}/")
    
    # 結果テーブル作成
    results_table = []
    for exp in kernel_experiments:
        results_table.append({
            'カーネルサイズ': f"{exp['kernel_size']}×{exp['kernel_size']}",
            '推定角度': f"{exp['result']['推定 角度(deg)']:.6f}",
            '推定スケール': f"{exp['result']['推定 スケール']:.6f}",
            '角度誤差': f"{exp['result']['角度誤差']:.6f}",
            'スケール誤差': f"{exp['result']['スケール誤差']:.6f}",
            '反復回数': int(exp['result']['反復回数'])
        })
    
    results_df = pd.DataFrame(results_table)
    results_df.to_csv(os.path.join(output_dir, 'kernel_size_results.csv'), 
                     index=False, encoding='utf-8-sig')

def plot_extreme_parameters(experiments, output_dir="visualization_results"):
    """
    実験4: 極端パラメータ実験
    """
    os.makedirs(output_dir, exist_ok=True)
    
    extreme_experiments = [exp for exp in experiments if exp['exp_type'] == 'extreme']
    if len(extreme_experiments) == 0:
        print("極端パラメータ実験のデータが見つかりません")
        return
    
    # 収束成功/失敗の判定
    for exp in extreme_experiments:
        exp['converged'] = int(exp['result']['反復回数']) < 999
    
    # 1. 収束過程の比較（新規追加）
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 角度収束過程
    ax1 = axes[0]
    for exp in extreme_experiments[:8]:  # 代表的な実験のみ表示
        history = exp['history']
        iterations = range(len(history))
        theta_values = history['回転角度θ（deg）']
        
        converged = exp['converged']
        style = '-' if converged else '--'
        alpha = 0.8 if converged else 0.5
        linewidth = 2 if converged else 1.5
        
        # ラベル作成
        if exp['true_scale'] != 1.5:
            label = f"s_true={exp['true_scale']:.1f}"
        elif exp['true_angle'] != 30.0:
            label = f"θ_true={exp['true_angle']:.0f}°"
        else:
            init_s, init_t = exp['init_values']
            label = f"init: s={init_s:.1f}, θ={init_t:.0f}°"
        
        ax1.plot(iterations, theta_values, style, 
                label=label, linewidth=linewidth, alpha=alpha)
    
    ax1.axhline(y=30.0, color='red', linestyle=':', linewidth=2, label='基準値 (30°)')
    ax1.set_xlabel('反復回数')
    ax1.set_ylabel('推定角度θ (度)')
    ax1.set_title('極端パラメータ：角度収束過程')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # スケール収束過程
    ax2 = axes[1]
    for exp in extreme_experiments[:8]:  # 代表的な実験のみ表示
        history = exp['history']
        iterations = range(len(history))
        scale_values = history['スケールs']
        
        converged = exp['converged']
        style = '-' if converged else '--'
        alpha = 0.8 if converged else 0.5
        linewidth = 2 if converged else 1.5
        
        # ラベル作成
        if exp['true_scale'] != 1.5:
            label = f"s_true={exp['true_scale']:.1f}"
        elif exp['true_angle'] != 30.0:
            label = f"θ_true={exp['true_angle']:.0f}°"
        else:
            init_s, init_t = exp['init_values']
            label = f"init: s={init_s:.1f}, θ={init_t:.0f}°"
        
        ax2.plot(iterations, scale_values, style, 
                label=label, linewidth=linewidth, alpha=alpha)
    
    ax2.axhline(y=1.5, color='red', linestyle=':', linewidth=2, label='基準値 (1.5)')
    ax2.set_xlabel('反復回数')
    ax2.set_ylabel('推定スケールs')
    ax2.set_title('極端パラメータ：スケール収束過程')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extreme_convergence_process.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 真値vs収束成功率のマップ
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # スケール真値の影響
    scale_true_values = sorted(list(set([exp['true_scale'] for exp in extreme_experiments 
                                       if exp['true_scale'] is not None])))
    scale_success_rates = []
    
    for scale_val in scale_true_values:
        scale_exps = [exp for exp in extreme_experiments if exp['true_scale'] == scale_val]
        if scale_exps:
            success_rate = sum([exp['converged'] for exp in scale_exps]) / len(scale_exps)
            scale_success_rates.append(success_rate)
        else:
            scale_success_rates.append(0)
    
    ax1 = axes[0]
    ax1.bar(range(len(scale_true_values)), scale_success_rates, alpha=0.8)
    ax1.set_xlabel('真値スケール')
    ax1.set_ylabel('収束成功率')
    ax1.set_title('スケール真値vs収束成功率')
    ax1.set_xticks(range(len(scale_true_values)))
    ax1.set_xticklabels([f'{s:.1f}' for s in scale_true_values])
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # 角度真値の影響
    angle_true_values = sorted(list(set([exp['true_angle'] for exp in extreme_experiments 
                                       if exp['true_angle'] is not None and exp['true_scale'] == 1.5])))
    angle_success_rates = []
    
    for angle_val in angle_true_values:
        angle_exps = [exp for exp in extreme_experiments 
                     if exp['true_angle'] == angle_val and exp['true_scale'] == 1.5]
        if angle_exps:
            success_rate = sum([exp['converged'] for exp in angle_exps]) / len(angle_exps)
            angle_success_rates.append(success_rate)
        else:
            angle_success_rates.append(0)
    
    ax2 = axes[1]
    ax2.bar(range(len(angle_true_values)), angle_success_rates, alpha=0.8, color='orange')
    ax2.set_xlabel('真値角度 (度)')
    ax2.set_ylabel('収束成功率')
    ax2.set_title('角度真値vs収束成功率')
    ax2.set_xticks(range(len(angle_true_values)))
    ax2.set_xticklabels([f'{a:.0f}' for a in angle_true_values])
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extreme_convergence_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 推定誤差の分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    converged_exps = [exp for exp in extreme_experiments if exp['converged']]
    diverged_exps = [exp for exp in extreme_experiments if not exp['converged']]
    
    if converged_exps:
        conv_angle_errors = [exp['result']['角度誤差'] for exp in converged_exps]
        conv_scale_errors = [exp['result']['スケール誤差'] for exp in converged_exps]
    else:
        conv_angle_errors = []
        conv_scale_errors = []
    
    if diverged_exps:
        div_angle_errors = [exp['result']['角度誤差'] for exp in diverged_exps]
        div_scale_errors = [exp['result']['スケール誤差'] for exp in diverged_exps]
    else:
        div_angle_errors = []
        div_scale_errors = []
    
    # 角度誤差分布
    ax1 = axes[0]
    if conv_angle_errors:
        ax1.hist(conv_angle_errors, bins=10, alpha=0.7, label='収束', color='blue')
    if div_angle_errors:
        ax1.hist(div_angle_errors, bins=10, alpha=0.7, label='非収束', color='red')
    ax1.set_xlabel('角度推定誤差 (度)')
    ax1.set_ylabel('頻度')
    ax1.set_title('角度推定誤差の分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # スケール誤差分布
    ax2 = axes[1]
    if conv_scale_errors:
        ax2.hist(conv_scale_errors, bins=10, alpha=0.7, label='収束', color='blue')
    if div_scale_errors:
        ax2.hist(div_scale_errors, bins=10, alpha=0.7, label='非収束', color='red')
    ax2.set_xlabel('スケール推定誤差')
    ax2.set_ylabel('頻度')
    ax2.set_title('スケール推定誤差の分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extreme_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"極端パラメータ分析グラフを保存: {output_dir}/")
    
    # 結果テーブル作成
    results_table = []
    for exp in extreme_experiments:
        results_table.append({
            '真値スケール': exp['true_scale'] if exp['true_scale'] else '-',
            '真値角度': exp['true_angle'] if exp['true_angle'] else '-',
            '初期値': f"s={exp['init_values'][0]:.1f}, θ={exp['init_values'][1]:.0f}" if exp['init_values'] else '-',
            '推定角度': f"{exp['result']['推定 角度(deg)']:.3f}",
            '推定スケール': f"{exp['result']['推定 スケール']:.3f}",
            '角度誤差': f"{exp['result']['角度誤差']:.3f}",
            'スケール誤差': f"{exp['result']['スケール誤差']:.3f}",
            '反復回数': int(exp['result']['反復回数']),
            '収束判定': '収束' if exp['converged'] else '非収束'
        })
    
    results_df = pd.DataFrame(results_table)
    results_df.to_csv(os.path.join(output_dir, 'extreme_parameters_results.csv'), 
                     index=False, encoding='utf-8-sig')

def create_summary_report(experiments, output_dir="visualization_results"):
    """
    統合サマリーレポート作成
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_text = []
    summary_text.append("=" * 80)
    summary_text.append("実験3・4 結果サマリーレポート")
    summary_text.append("=" * 80)
    
    # 実験3のサマリー
    kernel_experiments = [exp for exp in experiments if exp['exp_type'] == 'kernel']
    if kernel_experiments:
        summary_text.append("\n【実験3：カーネルサイズの影響】")
        summary_text.append(f"実験数: {len(kernel_experiments)}")
        
        best_accuracy = min(kernel_experiments, key=lambda x: x['result']['角度誤差'])
        fastest_conv = min(kernel_experiments, key=lambda x: x['result']['反復回数'])
        
        summary_text.append(f"最高精度: カーネル{best_accuracy['kernel_size']}×{best_accuracy['kernel_size']}, "
                          f"角度誤差={best_accuracy['result']['角度誤差']:.6f}°")
        summary_text.append(f"最速収束: カーネル{fastest_conv['kernel_size']}×{fastest_conv['kernel_size']}, "
                          f"{fastest_conv['result']['反復回数']}回")
    
    # 実験4のサマリー
    extreme_experiments = [exp for exp in experiments if exp['exp_type'] == 'extreme']
    if extreme_experiments:
        summary_text.append("\n【実験4：極端パラメータの影響】")
        summary_text.append(f"実験数: {len(extreme_experiments)}")
        
        converged_count = sum([int(exp['result']['反復回数']) < 999 for exp in extreme_experiments])
        summary_text.append(f"収束実験: {converged_count}/{len(extreme_experiments)} "
                          f"({converged_count/len(extreme_experiments)*100:.1f}%)")
        
        # 極値の特定
        extreme_successful = [exp for exp in extreme_experiments if int(exp['result']['反復回数']) < 999]
        if extreme_successful:
            max_scale = max([exp['true_scale'] for exp in extreme_successful if exp['true_scale']])
            max_angle = max([exp['true_angle'] for exp in extreme_successful if exp['true_angle']])
            summary_text.append(f"収束可能な最大スケール: {max_scale}")
            summary_text.append(f"収束可能な最大角度: {max_angle}°")
    
    # テキストファイルに保存
    summary_file = os.path.join(output_dir, "experiments_3_4_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_text))
    
    print(f"統合サマリーを保存: {summary_file}")

def main():
    """
    メイン処理
    """
    print("=== 実験3・4 可視化・分析開始 ===")
    
    # データ読み込み（複数ディレクトリに対応）
    base_dirs = [
        "実験3カーネルサイズ",
        "実験4極端パラメータ",
        "experiment3_kernel",
        "experiment4_extreme"
    ]
    
    experiments = load_experiment_data(base_dirs)
    
    if len(experiments) == 0:
        print("実験データが見つかりません")
        return
    
    print(f"\n読み込み完了: {len(experiments)}件の実験データ")
    
    # 実験3の可視化
    plot_kernel_size_comparison(experiments)
    
    # 実験4の可視化
    plot_extreme_parameters(experiments)
    
    # 統合サマリー
    create_summary_report(experiments)
    
    print("\n=== 可視化・分析完了 ===")
    print("以下のファイルが生成されました:")
    print("- visualization_results/kernel_size_convergence.png")
    print("- visualization_results/kernel_size_performance.png")
    print("- visualization_results/extreme_convergence_rates.png")
    print("- visualization_results/extreme_error_distribution.png")
    print("- visualization_results/kernel_size_results.csv")
    print("- visualization_results/extreme_parameters_results.csv")
    print("- visualization_results/experiments_3_4_summary.txt")

if __name__ == "__main__":
    main()