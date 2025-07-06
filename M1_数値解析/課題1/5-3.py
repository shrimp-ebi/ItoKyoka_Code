# 最小2乗法・最尤推定法・KCR下界に基づくRMS誤差評価プログラム（修正版）

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.linalg import eigh

# --- データ生成関連関数 ---
def generate_ellipse_points(N=100):
    """楕円上の点列を生成"""
    theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)
    x = 300 * np.cos(theta)
    y = 200 * np.sin(theta)
    return x, y

def add_noise(x, y, sigma):
    """ガウスノイズを追加"""
    x_noise = np.random.normal(0, sigma, len(x))
    y_noise = np.random.normal(0, sigma, len(y))
    return x + x_noise, y + y_noise

# --- 楕円パラメータ推定 ---
def least_squares_ellipse(x, y):
    """最小2乗法による楕円パラメータ推定"""
    xi = np.column_stack([x**2, 2*x*y, y**2, 2*x, 2*y, np.ones(len(x))])
    M = xi.T @ xi
    _, eigenvectors = eigh(M)
    u = eigenvectors[:, 0]
    return u

def covariance_matrix_xi(x, y, sigma):
    """ξベクトルの共分散行列を計算（修正版）"""
    # ∂ξ/∂x = [2x, 2y, 0, 2, 0, 0]
    # ∂ξ/∂y = [0, 2x, 2y, 0, 2, 0]
    dxi_dx = np.array([2*x, 2*y, 0, 2, 0, 0])
    dxi_dy = np.array([0, 2*x, 2*y, 0, 2, 0])
    
    # V[ξ] = σ²(∂ξ/∂x)(∂ξ/∂x)ᵀ + σ²(∂ξ/∂y)(∂ξ/∂y)ᵀ
    V = sigma**2 * (np.outer(dxi_dx, dxi_dx) + np.outer(dxi_dy, dxi_dy))
    return V

def maximum_likelihood_ellipse(x, y, sigma, max_iter=50, tol=1e-6):
    """最尤推定法による楕円パラメータ推定"""
    N = len(x)
    u = least_squares_ellipse(x, y)
    u = u / np.linalg.norm(u)

    for _ in range(max_iter):
        M = np.zeros((6, 6))
        L = np.zeros((6, 6))
        for i in range(N):
            xi = np.array([x[i]**2, 2*x[i]*y[i], y[i]**2, 2*x[i], 2*y[i], 1])
            V_xi = covariance_matrix_xi(x[i], y[i], sigma)
            uVu = u.T @ V_xi @ u
            if uVu > 1e-12:
                M += np.outer(xi, xi) / uVu
                L += (xi.T @ u)**2 * V_xi / (uVu**2)
        try:
            _, eigenvectors = eigh(M - L)
            u_new = eigenvectors[:, 0]
            u_new = u_new / np.linalg.norm(u_new)
        except:
            return u
        if np.linalg.norm(u_new - u) < tol:
            return u_new
        u = u_new
    return u

# --- RMS誤差計算 ---
def calculate_rms_error(u_estimates, u_true):
    """RMS誤差を計算"""
    u_true = u_true / np.linalg.norm(u_true)
    rms_total = 0
    for u in u_estimates:
        u = u / np.linalg.norm(u)
        # 符号を合わせる
        if np.dot(u, u_true) < 0:
            u = -u
        # 接空間への射影
        delta_u = (np.eye(6) - np.outer(u_true, u_true)) @ u
        rms_total += np.linalg.norm(delta_u)**2
    return np.sqrt(rms_total / len(u_estimates))

# --- KCR下界計算 ---
def calculate_kcr_lower_bound(x, y, u_true, sigma):
    """KCR下界を計算"""
    N = len(x)
    M = np.zeros((6, 6))
    for i in range(N):
        xi = np.array([x[i]**2, 2*x[i]*y[i], y[i]**2, 2*x[i], 2*y[i], 1])
        V_xi = covariance_matrix_xi(x[i], y[i], sigma)
        uVu = u_true.T @ V_xi @ u_true
        if uVu > 1e-12:
            M += np.outer(xi, xi) / uVu
    
    # Mの固有値を計算
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.sort(eigvals)[::-1]  # 大きい順にソート
    
    # 自由度5（6次元から制約1を引く）
    eigvals = eigvals[:5]
    
    if np.any(eigvals <= 0):
        return np.nan
    
    # KCR下界 = sqrt(Σ(1/λᵢ))
    return np.sqrt(np.sum(1 / eigvals))

# --- メイン処理 ---
def main():
    # パラメータ設定
    N = 100
    sigma_values = np.arange(0.1, 3.1, 0.1)
    num_trials = 1000
    
    # 真の楕円パラメータ
    u_true = np.array([1, 0, (300/200)**2, 0, 0, -300**2])
    u_true /= np.linalg.norm(u_true)

    # 結果を格納するリスト
    rms_ls = []
    rms_mle = []
    kcr_bounds = []
    
    # ノイズなしの真の点列を生成（KCR計算用）
    x_true, y_true = generate_ellipse_points(N)
    
    # σ=1でのKCR基準値を計算
    kcr_base = calculate_kcr_lower_bound(x_true, y_true, u_true, 1.0)
    print(f"KCR base value (σ=1): {kcr_base:.4e}")

    # 各ノイズレベルで計算
    for idx, sigma in enumerate(sigma_values):
        print(f"\n[σ = {sigma:.1f}] {idx+1}/{len(sigma_values)} 計算中...")
        u_ls_list = []
        u_mle_list = []
        
        # Monte Carlo試行
        for trial in range(num_trials):
            if trial % 200 == 0:
                print(f"  trial {trial}/{num_trials}...")
            
            # ノイズを加えたデータを生成
            x_noisy, y_noisy = add_noise(x_true, y_true, sigma)
            
            # パラメータ推定
            u_ls = least_squares_ellipse(x_noisy, y_noisy)
            u_mle = maximum_likelihood_ellipse(x_noisy, y_noisy, sigma)
            
            u_ls_list.append(u_ls)
            u_mle_list.append(u_mle)
        
        # RMS誤差を計算
        rms_ls.append(calculate_rms_error(u_ls_list, u_true))
        rms_mle.append(calculate_rms_error(u_mle_list, u_true))
        
        # KCR下界を計算（σに比例）
        kcr_bound = kcr_base * sigma
        kcr_bounds.append(kcr_bound)
        
        print(f"  LSM RMS = {rms_ls[-1]:.4e}, MLE RMS = {rms_mle[-1]:.4e}, KCR = {kcr_bound:.4e}")

    # --- グラフ描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, rms_ls, 'r-o', label='LSM', markersize=6)
    plt.plot(sigma_values, rms_mle, 'b-s', label='MLE', markersize=6)
    plt.plot(sigma_values, kcr_bounds, 'g-^', label='KCR Lower Bound', markersize=6)
    
    plt.xlabel('Noise Level $\sigma$', fontsize=12)
    plt.ylabel('RMS Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.title('RMS Error and KCR Bound vs. Noise Level', fontsize=14)

    
    plt.tight_layout()
    plt.savefig('5-3.png', dpi=300)
    print("\n図を保存しました：5-3.png")
    plt.show()

if __name__ == '__main__':
    main()