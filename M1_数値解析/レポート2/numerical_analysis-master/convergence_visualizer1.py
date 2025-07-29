# -*- coding: utf-8 -*-
"""
修正版：実験設計に合わせた収束過程可視化
角度固定実験とスケール固定実験を分離して表示
"""
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

def extract_initial_values_from_folder(folder_name):
    """
    フォルダ名から初期値を抽出
    例: 'Fuji_true_s1.5_t30.0_init_s1.0_t0.0' -> (1.0, 0.0)
    """
    print(f"    フォルダ名解析: {folder_name}")
    
    # init_s以降の部分を取得
    if 'init_s' not in folder_name:
        print(f"    'init_s'が見つかりません")
        return None, None
    
    try:
        # init_s以降を分割
        init_part = folder_name.split('init_s')[1]  # "1.0_t0.0" など
        print(f"    init_part: {init_part}")
        
        # _tで分割してスケールと角度を取得
        if '_t' in init_part:
            scale_str = init_part.split('_t')[0]  # "1.0"
            theta_str = init_part.split('_t')[1]  # "0.0"
            
            scale_init = float(scale_str)
            theta_init = float(theta_str)
            
            print(f"    抽出成功: scale={scale_init}, theta={theta_init}")
            return scale_init, theta_init
        else:
            print(f"    '_t'が見つかりません")
            return None, None
            
    except Exception as e:
        print(f"    抽出エラー: {e}")
        return None, None

def load_experiment_data(base_dir="output/実験1初期値変更"):
    """
    実験データを読み込み（全ての実験フォルダから）
    """
    experiments = []
    
    if not os.path.exists(base_dir):
        print(f"ディレクトリが見つかりません: {base_dir}")
        return experiments
    
    # より広範囲に検索
    search_patterns = [
        "Fuji_true_s1.5_t30.0_init_*",
    ]
    
    all_folders = []
    for pattern in search_patterns:
        folders = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        all_folders.extend(folders)
    
    # 重複削除
    all_folders = list(set(all_folders))
    
    print(f"検索パターン: {search_patterns}")
    print(f"見つかったフォルダ数: {len(all_folders)}")
    
    for folder in all_folders:
        folder_name = os.path.basename(folder)
        print(f"処理中: {folder_name}")
        scale_init, theta_init = extract_initial_values_from_folder(folder_name)
        
        if scale_init is None or theta_init is None:
            print(f"  初期値抽出失敗: scale={scale_init}, theta={theta_init}")
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
                'scale_init': scale_init,
                'theta_init': theta_init,
                'result': result,
                'history': history_df,
                'folder_name': folder_name,
                'folder_path': folder
            }
            
            experiments.append(experiment)
            print(f"  読み込み成功: scale={scale_init}, theta={theta_init}")
            
        except Exception as e:
            print(f"  エラー {folder}: {e}")
    
    return experiments

def plot_scale_init_experiments(experiments, output_dir="convergence_plots"):
    """
    スケール初期値実験（角度固定）の収束過程プロット
    θ₀=30°固定、スケール初期値変化
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # スケール初期値実験のデータを抽出（theta_init=30で固定）
    scale_experiments = [exp for exp in experiments if exp['theta_init'] == 30.0]
    scale_experiments.sort(key=lambda x: x['scale_init'])
    
    print(f"スケール初期値実験データ数: {len(scale_experiments)}")
    for exp in scale_experiments:
        print(f"  s_init={exp['scale_init']}, θ_init={exp['theta_init']}")
    
    if len(scale_experiments) == 0:
        print("スケール初期値実験のデータが見つかりません")
        return
    
    # 1. 角度収束過程
    plt.figure(figsize=(12, 8))
    
    for exp in scale_experiments:
        history = exp['history']
        iterations = range(len(history))
        theta_values = history['回転角度θ（deg）']
        
        converged = int(exp['result']['反復回数']) < 999
        style = '-' if converged else '--'
        alpha = 0.8 if converged else 0.5
        linewidth = 2 if converged else 1.5
        
        label = f"s₀={exp['scale_init']}, θ₀={exp['theta_init']:.0f}°"
        plt.plot(iterations, theta_values, style, 
                label=label, linewidth=linewidth, alpha=alpha)
    
    plt.axhline(y=30.0, color='red', linestyle=':', linewidth=3, label='真値 (30°)')
    plt.xlabel('反復回数')
    plt.ylabel('推定角度θ (度)')
    plt.title('全実験における角度収束過程')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_init_theta_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. スケール収束過程
    plt.figure(figsize=(12, 8))
    
    for exp in scale_experiments:
        history = exp['history']
        iterations = range(len(history))
        scale_values = history['スケールs']
        
        converged = int(exp['result']['反復回数']) < 999
        style = '-' if converged else '--'
        alpha = 0.8 if converged else 0.5
        linewidth = 2 if converged else 1.5
        
        label = f"s₀={exp['scale_init']}, θ₀={exp['theta_init']:.0f}°"
        plt.plot(iterations, scale_values, style, 
                label=label, linewidth=linewidth, alpha=alpha)
    
    plt.axhline(y=1.5, color='red', linestyle=':', linewidth=3, label='真値 (1.5)')
    plt.xlabel('反復回数')
    plt.ylabel('推定スケールs')
    plt.title('全実験におけるスケール収束過程')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_init_scale_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"スケール初期値実験プロットを保存: {output_dir}/scale_init_theta_convergence.png")
    print(f"スケール初期値実験プロットを保存: {output_dir}/scale_init_scale_convergence.png")

def plot_theta_init_experiments(experiments, output_dir="convergence_plots"):
    """
    角度初期値実験（スケール固定）の収束過程プロット
    s₀=1.5固定、角度初期値変化
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 角度初期値実験のデータを抽出（scale_init=1.5で固定）
    theta_experiments = [exp for exp in experiments if exp['scale_init'] == 1.5]
    theta_experiments.sort(key=lambda x: x['theta_init'])
    
    print(f"角度初期値実験データ数: {len(theta_experiments)}")
    for exp in theta_experiments:
        print(f"  s_init={exp['scale_init']}, θ_init={exp['theta_init']}")
    
    if len(theta_experiments) == 0:
        print("角度初期値実験のデータが見つかりません")
        return
    
    # 1. 角度収束過程
    plt.figure(figsize=(12, 8))
    
    for exp in theta_experiments:
        history = exp['history']
        iterations = range(len(history))
        theta_values = history['回転角度θ（deg）']
        
        converged = int(exp['result']['反復回数']) < 999
        style = '-' if converged else '--'
        alpha = 0.8 if converged else 0.5
        linewidth = 2 if converged else 1.5
        
        label = f"s₀={exp['scale_init']}, θ₀={exp['theta_init']:.0f}°"
        plt.plot(iterations, theta_values, style, 
                label=label, linewidth=linewidth, alpha=alpha)
    
    plt.axhline(y=30.0, color='red', linestyle=':', linewidth=3, label='真値 (30°)')
    plt.xlabel('反復回数')
    plt.ylabel('推定角度θ (度)')
    plt.title('全実験における角度収束過程')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theta_init_theta_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. スケール収束過程
    plt.figure(figsize=(12, 8))
    
    for exp in theta_experiments:
        history = exp['history']
        iterations = range(len(history))
        scale_values = history['スケールs']
        
        converged = int(exp['result']['反復回数']) < 999
        style = '-' if converged else '--'
        alpha = 0.8 if converged else 0.5
        linewidth = 2 if converged else 1.5
        
        label = f"s₀={exp['scale_init']}, θ₀={exp['theta_init']:.0f}°"
        plt.plot(iterations, scale_values, style, 
                label=label, linewidth=linewidth, alpha=alpha)
    
    plt.axhline(y=1.5, color='red', linestyle=':', linewidth=3, label='真値 (1.5)')
    plt.xlabel('反復回数')
    plt.ylabel('推定スケールs')
    plt.title('全実験におけるスケール収束過程')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theta_init_scale_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"角度初期値実験プロットを保存: {output_dir}/theta_init_theta_convergence.png")
    print(f"角度初期値実験プロットを保存: {output_dir}/theta_init_scale_convergence.png")

def create_summary_statistics(experiments, output_dir="convergence_plots"):
    """
    統計サマリーの作成とテキストファイル出力
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== 全実験データサマリー ===")
    print(f"総実験数: {len(experiments)}")
    
    # 全実験を初期値でソート
    experiments_sorted = sorted(experiments, key=lambda x: (x['scale_init'], x['theta_init']))
    
    # テキストファイルに出力するための文字列を作成
    summary_text = []
    summary_text.append("=" * 80)
    summary_text.append("実験1：初期値変更実験 結果サマリー")
    summary_text.append("=" * 80)
    summary_text.append(f"総実験数: {len(experiments)}")
    summary_text.append("")
    
    print("\n【全実験結果】")
    print("scale_init | theta_init | 推定角度 | 推定スケール | 角度誤差 | スケール誤差 | 反復回数 | 収束判定")
    print("-" * 100)
    
    summary_text.append("【全実験結果一覧】")
    summary_text.append("scale_init | theta_init | 推定角度 | 推定スケール | 角度誤差 | スケール誤差 | 反復回数 | 収束判定")
    summary_text.append("-" * 100)
    
    for exp in experiments_sorted:
        iterations = int(exp['result']['反復回数'])
        converged = "収束" if iterations < 999 else "非収束"
        line = (f"{exp['scale_init']:8.1f} | {exp['theta_init']:8.1f} | {exp['result']['推定 角度(deg)']:8.3f} | "
                f"{exp['result']['推定 スケール']:10.6f} | {exp['result']['角度誤差']:8.6f} | "
                f"{exp['result']['スケール誤差']:11.6f} | {iterations:6d} | {converged}")
        print(line)
        summary_text.append(line)
    
    # スケール初期値実験
    scale_exp = [exp for exp in experiments if exp['theta_init'] == 30.0]
    scale_exp.sort(key=lambda x: x['scale_init'])
    
    # 角度初期値実験  
    theta_exp = [exp for exp in experiments if exp['scale_init'] == 1.5]
    theta_exp.sort(key=lambda x: x['theta_init'])
    
    print("\n=== 分類別統計 ===")
    summary_text.append("")
    summary_text.append("=" * 50)
    summary_text.append("分類別統計")
    summary_text.append("=" * 50)
    
    print("\n【スケール初期値実験（角度固定θ₀=30°）】")
    summary_text.append("")
    summary_text.append("【スケール初期値実験（角度固定θ₀=30°）】")
    summary_text.append("真値: スケール=1.5, 角度=30.0度")
    summary_text.append("")
    
    for exp in scale_exp:
        iterations = int(exp['result']['反復回数'])
        converged = "収束" if iterations < 999 else "非収束"
        line = (f"s_init={exp['scale_init']:4.1f}: {iterations:3d}回 ({converged}), "
                f"角度誤差={exp['result']['角度誤差']:.6f}°, "
                f"スケール誤差={exp['result']['スケール誤差']:.6f}")
        print(line)
        summary_text.append(line)
    
    print("\n【角度初期値実験（スケール固定s₀=1.5）】")
    summary_text.append("")
    summary_text.append("【角度初期値実験（スケール固定s₀=1.5）】")
    summary_text.append("真値: スケール=1.5, 角度=30.0度")
    summary_text.append("")
    
    for exp in theta_exp:
        iterations = int(exp['result']['反復回数'])
        converged = "収束" if iterations < 999 else "非収束"
        line = (f"θ_init={exp['theta_init']:4.0f}°: {iterations:3d}回 ({converged}), "
                f"角度誤差={exp['result']['角度誤差']:.6f}°, "
                f"スケール誤差={exp['result']['スケール誤差']:.6f}")
        print(line)
        summary_text.append(line)
    
    # 収束率の計算
    total_converged = len([exp for exp in experiments if int(exp['result']['反復回数']) < 999])
    scale_converged = len([exp for exp in scale_exp if int(exp['result']['反復回数']) < 999]) if scale_exp else 0
    theta_converged = len([exp for exp in theta_exp if int(exp['result']['反復回数']) < 999]) if theta_exp else 0
    
    print(f"\n【収束率】")
    print(f"全実験: {total_converged}/{len(experiments)} ({total_converged/len(experiments)*100:.1f}%)")
    if len(scale_exp) > 0:
        print(f"スケール初期値実験: {scale_converged}/{len(scale_exp)} ({scale_converged/len(scale_exp)*100:.1f}%)")
    if len(theta_exp) > 0:
        print(f"角度初期値実験: {theta_converged}/{len(theta_exp)} ({theta_converged/len(theta_exp)*100:.1f}%)")
    
    summary_text.append("")
    summary_text.append("【収束率】")
    summary_text.append(f"全実験: {total_converged}/{len(experiments)} ({total_converged/len(experiments)*100:.1f}%)")
    if len(scale_exp) > 0:
        summary_text.append(f"スケール初期値実験: {scale_converged}/{len(scale_exp)} ({scale_converged/len(scale_exp)*100:.1f}%)")
    if len(theta_exp) > 0:
        summary_text.append(f"角度初期値実験: {theta_converged}/{len(theta_exp)} ({theta_converged/len(theta_exp)*100:.1f}%)")
    
    # 詳細分析の追加
    summary_text.append("")
    summary_text.append("=" * 50)
    summary_text.append("詳細分析")
    summary_text.append("=" * 50)
    
    # スケール初期値実験の分析
    if len(scale_exp) > 0:
        summary_text.append("")
        summary_text.append("【スケール初期値実験の特徴】")
        
        # 収束したものと非収束のものを分類
        scale_converged_exp = [exp for exp in scale_exp if int(exp['result']['反復回数']) < 999]
        scale_diverged_exp = [exp for exp in scale_exp if int(exp['result']['反復回数']) >= 999]
        
        if scale_converged_exp:
            summary_text.append(f"収束実験数: {len(scale_converged_exp)}")
            min_iter = min([int(exp['result']['反復回数']) for exp in scale_converged_exp])
            max_iter = max([int(exp['result']['反復回数']) for exp in scale_converged_exp])
            summary_text.append(f"収束反復回数範囲: {min_iter}～{max_iter}回")
            
            # 最高精度
            best_scale_exp = min(scale_converged_exp, key=lambda x: x['result']['スケール誤差'])
            summary_text.append(f"最高スケール精度: s_init={best_scale_exp['scale_init']}, 誤差={best_scale_exp['result']['スケール誤差']:.6f}")
            
            best_angle_exp = min(scale_converged_exp, key=lambda x: x['result']['角度誤差'])
            summary_text.append(f"最高角度精度: s_init={best_angle_exp['scale_init']}, 誤差={best_angle_exp['result']['角度誤差']:.6f}°")
        
        if scale_diverged_exp:
            summary_text.append(f"非収束実験数: {len(scale_diverged_exp)}")
            for exp in scale_diverged_exp:
                summary_text.append(f"  s_init={exp['scale_init']}: 推定値 s={exp['result']['推定 スケール']:.3f}, θ={exp['result']['推定 角度(deg)']:.3f}°")
    
    # 角度初期値実験の分析
    if len(theta_exp) > 0:
        summary_text.append("")
        summary_text.append("【角度初期値実験の特徴】")
        
        # 収束したものと非収束のものを分類
        theta_converged_exp = [exp for exp in theta_exp if int(exp['result']['反復回数']) < 999]
        theta_diverged_exp = [exp for exp in theta_exp if int(exp['result']['反復回数']) >= 999]
        
        if theta_converged_exp:
            summary_text.append(f"収束実験数: {len(theta_converged_exp)}")
            min_iter = min([int(exp['result']['反復回数']) for exp in theta_converged_exp])
            max_iter = max([int(exp['result']['反復回数']) for exp in theta_converged_exp])
            summary_text.append(f"収束反復回数範囲: {min_iter}～{max_iter}回")
            
            # 最速収束
            fastest_exp = min(theta_converged_exp, key=lambda x: int(x['result']['反復回数']))
            summary_text.append(f"最速収束: θ_init={fastest_exp['theta_init']:.0f}°, {int(fastest_exp['result']['反復回数'])}回")
            
            # 最高精度
            best_angle_exp = min(theta_converged_exp, key=lambda x: x['result']['角度誤差'])
            summary_text.append(f"最高角度精度: θ_init={best_angle_exp['theta_init']:.0f}°, 誤差={best_angle_exp['result']['角度誤差']:.6f}°")
        
        if theta_diverged_exp:
            summary_text.append(f"非収束実験数: {len(theta_diverged_exp)}")
            for exp in theta_diverged_exp:
                summary_text.append(f"  θ_init={exp['theta_init']:.0f}°: 推定値 s={exp['result']['推定 スケール']:.3f}, θ={exp['result']['推定 角度(deg)']:.3f}°")
    
    # 収束方向性の分析
    summary_text.append("")
    summary_text.append("【収束方向性の分析】")
    
    if len(scale_exp) > 0:
        summary_text.append("スケールパラメータ:")
        below_true = [exp for exp in scale_exp if exp['scale_init'] < 1.5]
        above_true = [exp for exp in scale_exp if exp['scale_init'] > 1.5]
        
        if below_true:
            below_converged = len([exp for exp in below_true if int(exp['result']['反復回数']) < 999])
            summary_text.append(f"  真値未満の初期値({len(below_true)}個): {below_converged}個収束 ({below_converged/len(below_true)*100:.1f}%)")
        
        if above_true:
            above_converged = len([exp for exp in above_true if int(exp['result']['反復回数']) < 999])
            summary_text.append(f"  真値超過の初期値({len(above_true)}個): {above_converged}個収束 ({above_converged/len(above_true)*100:.1f}%)")
    
    if len(theta_exp) > 0:
        summary_text.append("角度パラメータ:")
        below_true = [exp for exp in theta_exp if exp['theta_init'] < 30.0]
        above_true = [exp for exp in theta_exp if exp['theta_init'] > 30.0]
        
        if below_true:
            below_converged = len([exp for exp in below_true if int(exp['result']['反復回数']) < 999])
            summary_text.append(f"  真値未満の初期値({len(below_true)}個): {below_converged}個収束 ({below_converged/len(below_true)*100:.1f}%)")
        
        if above_true:
            above_converged = len([exp for exp in above_true if int(exp['result']['反復回数']) < 999])
            summary_text.append(f"  真値超過の初期値({len(above_true)}個): {above_converged}個収束 ({above_converged/len(above_true)*100:.1f}%)")
    
    # テキストファイルに保存
    summary_file = os.path.join(output_dir, "experiment1_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_text))
    
    print(f"\n統計サマリーをファイルに保存: {summary_file}")
    
    # CSVファイルとしても保存
    csv_data = []
    for exp in experiments_sorted:
        csv_data.append({
            'scale_init': exp['scale_init'],
            'theta_init': exp['theta_init'],
            'estimated_angle': exp['result']['推定 角度(deg)'],
            'estimated_scale': exp['result']['推定 スケール'],
            'angle_error': exp['result']['角度誤差'],
            'scale_error': exp['result']['スケール誤差'],
            'iterations': int(exp['result']['反復回数']),
            'converged': int(exp['result']['反復回数']) < 999,
            'experiment_type': 'scale_init' if exp['theta_init'] == 30.0 else 'theta_init'
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_file = os.path.join(output_dir, "experiment1_results.csv")
    csv_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"結果データをCSVに保存: {csv_file}")

def main():
    """
    メイン処理
    """
    print("=== 実験設計に合わせた収束過程可視化開始 ===")
    
    # データ読み込み
    experiments = load_experiment_data()
    
    if len(experiments) == 0:
        print("実験データが見つかりません")
        return
    
    print(f"\n読み込み完了: {len(experiments)}件の実験データ")
    
    # 実験別の分離プロット
    plot_scale_init_experiments(experiments)  # 角度固定、スケール変化
    plot_theta_init_experiments(experiments)  # スケール固定、角度変化
    
    # 統計サマリー
    create_summary_statistics(experiments)
    
    print("\n=== 可視化完了 ===")
    print("以下のファイルが生成されました:")
    print("- convergence_plots/scale_init_theta_convergence.png (スケール実験：角度収束)")
    print("- convergence_plots/scale_init_scale_convergence.png (スケール実験：スケール収束)")
    print("- convergence_plots/theta_init_theta_convergence.png (角度実験：角度収束)")
    print("- convergence_plots/theta_init_scale_convergence.png (角度実験：スケール収束)")
    print("- convergence_plots/experiment1_summary.txt (詳細分析テキスト)")
    print("- convergence_plots/experiment1_results.csv (結果データCSV)")

if __name__ == "__main__":
    main()