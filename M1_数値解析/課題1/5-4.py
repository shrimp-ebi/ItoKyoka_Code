import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

def generate_ellipse_points(N=100):
    """楕円上の点列を生成"""
    theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)
    x = 300 * np.cos(theta)
    y = 200 * np.sin(theta)
    return x, y

def add_noise(x, y, sigma):
    """ガウシアンノイズを追加"""
    x_noise = np.random.normal(0, sigma, len(x))
    y_noise = np.random.normal(0, sigma, len(y))
    return x + x_noise, y + y_noise

def least_squares_ellipse(x, y):
    """最小2乗法による楕円当てはめ"""
    # データベクトルの構築
    xi = np.column_stack([x**2, 2*x*y, y**2, 2*x, 2*y, np.ones(len(x))])
    
    # 行列M = Σ(xi * xi^T)
    M = xi.T @ xi
    
    # 最小固有値に対応する固有ベクトルを求める
    eigenvalues, eigenvectors = eigh(M)
    u = eigenvectors[:, 0]  # 最小固有値に対応する固有ベクトル
    
    return u

def covariance_matrix_xi(x, y, sigma):
    """ξ空間での共分散行列を計算（修正版）"""
    # ∂ξ/∂x = [2x, 2y, 0, 2, 0, 0]
    # ∂ξ/∂y = [0, 2x, 2y, 0, 2, 0]
    dxi_dx = np.array([2*x, 2*y, 0, 2, 0, 0])
    dxi_dy = np.array([0, 2*x, 2*y, 0, 2, 0])
    
    # V[ξ] = σ²(∂ξ/∂x)(∂ξ/∂x)ᵀ + σ²(∂ξ/∂y)(∂ξ/∂y)ᵀ
    V = sigma**2 * (np.outer(dxi_dx, dxi_dx) + np.outer(dxi_dy, dxi_dy))
    return V

def maximum_likelihood_ellipse(x, y, sigma, max_iter=100, tol=1e-8):
    """最尤推定法による楕円当てはめ（反復解法）"""
    N = len(x)
    
    # 初期値として最小2乗解を使用
    u = least_squares_ellipse(x, y)
    u = u / np.linalg.norm(u)
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # 行列MとLを計算
        M = np.zeros((6, 6))
        L = np.zeros((6, 6))
        
        for i in range(N):
            xi = np.array([x[i]**2, 2*x[i]*y[i], y[i]**2, 2*x[i], 2*y[i], 1])
            V_xi = covariance_matrix_xi(x[i], y[i], sigma)
            
            # uとV_xiの積を計算
            u_V_u = u.T @ V_xi @ u
            
            if u_V_u > 1e-12:  # 数値的安定性のチェック
                xi_u_2 = (xi.T @ u)**2
                M += np.outer(xi, xi) / u_V_u
                L += xi_u_2 * V_xi / (u_V_u**2)
        
        # 一般固有値問題 (M - L)u = λu を解く
        try:
            eigenvalues, eigenvectors = eigh(M - L)
            # 最小固有値に対応する固有ベクトルを選択
            u = eigenvectors[:, 0]
            u = u / np.linalg.norm(u)
        except np.linalg.LinAlgError:
            break
        
        # 収束判定
        if np.linalg.norm(u - u_old) < tol or np.linalg.norm(u + u_old) < tol:
            break
    
    return u

def calculate_rms_error(u_estimates, u_true):
    """RMS誤差を計算（複数の推定値に対応）"""
    # 真値ベクトルを正規化
    u_true_norm = u_true / np.linalg.norm(u_true)
    
    rms_total = 0
    count = 0
    
    for u_est in u_estimates:
        # 推定値を正規化
        u_est_norm = u_est / np.linalg.norm(u_est)
        
        # 符号の調整（内積が負の場合は符号を反転）
        if np.dot(u_true_norm, u_est_norm) < 0:
            u_est_norm = -u_est_norm
        
        # 射影行列 P_u = I - u_true * u_true^T
        P_u = np.eye(6) - np.outer(u_true_norm, u_true_norm)
        
        # Δu = P_u * u_estimated
        delta_u = P_u @ u_est_norm
        
        # 二乗誤差を累積
        rms_total += np.linalg.norm(delta_u)**2
        count += 1
    
    # RMS誤差
    return np.sqrt(rms_total / count) if count > 0 else 0

def calculate_kcr_bound(x, y, sigma, u_true):
    """KCR下界を計算"""
    N = len(x)
    
    # 行列Mを計算
    M = np.zeros((6, 6))
    
    for i in range(N):
        xi = np.array([x[i]**2, 2*x[i]*y[i], y[i]**2, 2*x[i], 2*y[i], 1])
        V_xi = covariance_matrix_xi(x[i], y[i], sigma)
        
        # uとV_xiの積を計算
        u_V_u = u_true.T @ V_xi @ u_true
        
        if u_V_u > 1e-12:  # 数値的安定性のチェック
            M += np.outer(xi, xi) / u_V_u
    
    # Mの固有値を計算（降順）
    eigenvalues = np.linalg.eigvalsh(M)
    eigenvalues = np.sort(eigenvalues)[::-1]  # 降順にソート
    
    # モデルの自由度r=5（楕円は6パラメータ-1制約=5自由度）
    r = 5
    
    # 上位r個の固有値を使用
    eigenvalues = eigenvalues[:r]
    
    # KCR下界を計算
    if np.any(eigenvalues <= 1e-12):
        return np.nan
    
    D_KCR = np.sum(1.0 / eigenvalues)
    
    return np.sqrt(D_KCR)

def main():
    # パラメータ設定
    N = 100
    sigma_max = 2.0
    sigma_values = np.arange(0.1, sigma_max + 0.1, 0.1)
    num_trials = 1000
    
    # 真の楕円パラメータ（修正版）
    # 楕円の方程式: x²/300² + y²/200² = 1
    # これを ax² + 2bxy + cy² + 2dx + 2ey + f = 0 の形に変換
    # x²/300² + y²/200² - 1 = 0
    # => x² + (300/200)²y² - 300² = 0
    u_true = np.array([1, 0, (300/200)**2, 0, 0, -300**2])
    u_true = u_true / np.linalg.norm(u_true)
    
    # 結果を格納する配列
    rms_least_squares = np.zeros(len(sigma_values))
    rms_maximum_likelihood = np.zeros(len(sigma_values))
    kcr_bounds = np.zeros(len(sigma_values))
    
    print("Starting ellipse fitting experiment...")
    print(f"Noise level: {sigma_values[0]:.1f} to {sigma_values[-1]:.1f}")
    print(f"Number of trials: {num_trials}")
    print(f"True ellipse parameters (normalized): {u_true}")
    
    # 真の楕円点列を一度生成（KCR計算用）
    x_true_base, y_true_base = generate_ellipse_points(N)
    
    for i, sigma in enumerate(sigma_values):
        print(f"\nσ = {sigma:.1f} の処理中... ({i+1}/{len(sigma_values)})")
        
        u_ls_trials = []
        u_ml_trials = []
        
        # KCR下界を計算（各σに対して）
        kcr_bounds[i] = calculate_kcr_bound(x_true_base, y_true_base, sigma, u_true)
        
        for trial in range(num_trials):
            if trial % 200 == 0:
                print(f"  trial {trial}/{num_trials}...")
            
            # ノイズ付きデータの生成
            x_noisy, y_noisy = add_noise(x_true_base, y_true_base, sigma)
            
            try:
                # 最小2乗法
                u_ls = least_squares_ellipse(x_noisy, y_noisy)
                u_ls_trials.append(u_ls)
                
                # 最尤推定法
                u_ml = maximum_likelihood_ellipse(x_noisy, y_noisy, sigma)
                u_ml_trials.append(u_ml)
                
            except:
                continue
        
        # RMS誤差を計算
        if u_ls_trials:
            rms_least_squares[i] = calculate_rms_error(u_ls_trials, u_true)
        if u_ml_trials:
            rms_maximum_likelihood[i] = calculate_rms_error(u_ml_trials, u_true)
        
        print(f"  LSM RMS = {rms_least_squares[i]:.4e}")
        print(f"  MLE RMS = {rms_maximum_likelihood[i]:.4e}")
        print(f"  KCR Bound = {kcr_bounds[i]:.4e}")
    
    # 結果のプロット
    plt.figure(figsize=(12, 8))
    plt.plot(sigma_values, rms_least_squares, 'r-o', label='LSM', markersize=6)
    plt.plot(sigma_values, rms_maximum_likelihood, 'b-s', label='MLE', markersize=6)
    plt.plot(sigma_values, kcr_bounds, 'g-^', label='KCR Lower Bound', markersize=6)

    plt.xlabel('Noise Level σ', fontsize=12)
    plt.ylabel('RMS Error', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, sigma_max)
    
    # y軸の上限を調整
    max_val = max(np.max(rms_least_squares), np.max(rms_maximum_likelihood), np.max(kcr_bounds))
    plt.ylim(0, max_val * 1.1)
    
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    plt.title('RMS Error and KCR Bound vs. Noise Level', fontsize=14)
    plt.tight_layout()
    plt.savefig('5-4.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n図を保存しました: 5-4.png")

if __name__ == "__main__":
    main()