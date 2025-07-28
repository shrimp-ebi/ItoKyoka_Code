import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_fuji_experiments():
    """Fuji.jpg実験結果の詳細分析"""
    
    # 結果収集
    results = []
    csv_files = glob.glob("output/**/result_summary.csv", recursive=True)
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            if len(df) > 0:
                result = df.iloc[0].to_dict()
                folder_name = os.path.basename(os.path.dirname(csv_file))
                result['folder'] = folder_name
                
                # 実験タイプを推定
                if 'scale_init' in folder_name:
                    result['experiment_type'] = 'scale_init'
                elif 'theta_init' in folder_name:
                    result['experiment_type'] = 'theta_init'
                elif 'combined' in folder_name:
                    result['experiment_type'] = 'combined'
                elif 'extreme' in folder_name:
                    result['experiment_type'] = 'extreme'
                else:
                    result['experiment_type'] = 'other'
                
                results.append(result)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not results:
        print("結果ファイルが見つかりません")
        return
    
    df = pd.DataFrame(results)
    
    # 分析結果の表示
    print("=== 実験結果サマリー ===")
    print(f"総実験数: {len(df)}")
    print(f"平均反復回数: {df['反復回数'].mean():.1f}")
    print(f"最小反復回数: {df['反復回数'].min()}")
    print(f"最大反復回数: {df['反復回数'].max()}")
    print(f"平均角度誤差: {df['角度誤差'].mean():.6f}度")
    print(f"平均スケール誤差: {df['スケール誤差'].mean():.6f}")
    
    # 図の作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 初期値と反復回数の関係
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['初期値 スケール'], df['反復回数'], 
                         c=df['初期値 角度'], cmap='viridis', alpha=0.7, s=60)
    ax1.set_xlabel('Scale Initial Value')
    ax1.set_ylabel('Iterations to Convergence')
    ax1.set_title('Initial Scale vs Convergence Speed')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Initial Theta (deg)')
    
    # 2. 角度初期値と反復回数
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['初期値 角度'], df['反復回数'], 
                          c=df['初期値 スケール'], cmap='plasma', alpha=0.7, s=60)
    ax2.set_xlabel('Theta Initial Value (deg)')
    ax2.set_ylabel('Iterations to Convergence')
    ax2.set_title('Initial Theta vs Convergence Speed')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Initial Scale')
    
    # 3. 推定精度
    ax3 = axes[0, 2]
    ax3.scatter(df['角度誤差'], df['スケール誤差'], alpha=0.7, s=60)
    ax3.set_xlabel('Theta Error (deg)')
    ax3.set_ylabel('Scale Error')
    ax3.set_title('Estimation Accuracy')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    
    # 4. 初期値距離と収束性
    df['scale_distance'] = np.abs(df['初期値 スケール'] - df['真値 スケール'])
    df['theta_distance'] = np.abs(df['初期値 角度'] - df['真値 角度(deg)'])
    df['total_distance'] = np.sqrt(df['scale_distance']**2 + (df['theta_distance']/45)**2)
    
    ax4 = axes[1, 0]
    ax4.scatter(df['total_distance'], df['反復回数'], alpha=0.7, s=60)
    ax4.set_xlabel('Distance from True Value (normalized)')
    ax4.set_ylabel('Iterations to Convergence')
    ax4.set_title('Initial Distance vs Convergence')
    ax4.grid(True, alpha=0.3)
    
    # 5. ヒートマップ（スケール vs 角度の収束性）
    ax5 = axes[1, 1]
    pivot_iterations = df.pivot_table(
        values='反復回数', 
        index='初期値 角度', 
        columns='初期値 スケール', 
        aggfunc='mean'
    )
    if not pivot_iterations.empty:
        sns.heatmap(pivot_iterations, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax5)
        ax5.set_title('Convergence Heatmap (Iterations)')
        ax5.set_xlabel('Initial Scale')
        ax5.set_ylabel('Initial Theta (deg)')
    
    # 6. 収束履歴の例
    ax6 = axes[1, 2]
    # 最良と最悪の例を表示
    best_idx = df['反復回数'].idxmin()
    worst_idx = df['反復回数'].idxmax()
    
    # 収束履歴を読み込み（例）
    try:
        best_folder = df.loc[best_idx, 'folder']
        best_history = pd.read_csv(f"output/{best_folder}/history.csv")
        ax6.plot(best_history['theta_history'], label=f'Best (iter={df.loc[best_idx, "反復回数"]})', linewidth=2)
        
        worst_folder = df.loc[worst_idx, 'folder']
        worst_history = pd.read_csv(f"output/{worst_folder}/history.csv")
        ax6.plot(worst_history['theta_history'], label=f'Worst (iter={df.loc[worst_idx, "反復回数"]})', linewidth=2)
        
        ax6.axhline(y=5.0, color='r', linestyle='--', label='True Value')
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Theta (deg)')
        ax6.set_title('Convergence Examples')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    except:
        ax6.text(0.5, 0.5, 'History data\nnot available', 
                ha='center', va='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig('output/fuji_analysis_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 統計的分析
    print("\n=== 統計分析 ===")
    
    # 初期値カテゴリごとの分析
    print("\n初期値タイプ別統計:")
    for exp_type in df['experiment_type'].unique():
        subset = df[df['experiment_type'] == exp_type]
        print(f"{exp_type}: 平均反復回数 {subset['反復回数'].mean():.1f}, "
              f"成功率 {len(subset[subset['反復回数'] < 1000])/len(subset)*100:.1f}%")
    
    # 相関分析
    print(f"\n相関分析:")
    print(f"初期スケール距離 vs 反復回数: {df['scale_distance'].corr(df['反復回数']):.3f}")
    print(f"初期角度距離 vs 反復回数: {df['theta_distance'].corr(df['反復回数']):.3f}")
    print(f"総初期距離 vs 反復回数: {df['total_distance'].corr(df['反復回数']):.3f}")
    
    # 最適初期値の提案
    best_results = df.nsmallest(5, '反復回数')
    print(f"\n最も速く収束した5つの初期値:")
    for i, (_, row) in enumerate(best_results.iterrows(), 1):
        print(f"{i}. スケール={row['初期値 スケール']:.1f}, "
              f"角度={row['初期値 角度']:.1f}度 → {row['反復回数']}回")
    
    # 結果をCSVで保存
    df.to_csv('output/fuji_complete_analysis.csv', index=False, encoding='utf-8-sig')
    
    return df

if __name__ == "__main__":
    results = analyze_fuji_experiments()