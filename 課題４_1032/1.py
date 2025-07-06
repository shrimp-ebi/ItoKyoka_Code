import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

def generate_ellipse_points(N=100):
    """楕円上の点列を生成"""
    theta = np.linspace(-np.pi/4, -np.pi/4 + 11*np.pi/12, N)
    x = 300 * np.cos(theta)
    y = 200 * np.sin(theta)
    return x, y

def xi_vector(x, y):
    """楕円の一般形のベクトル"""
    return np.array([x**2, 2*x*y, y**2, 2*x, 2*y, 1])

def compute_covariance_matrix(x, y, sigma):
    """共分散行列V[ξ]の計算"""
    V = 4 * sigma**2 * np.array([
        [x**2, x*y, 0, x, 0, 0],
        [x*y, x**2 + y**2, x*y, y, x, 0],
        [0, x*y, y**2, 0, y, 0],
        [x, y, 0, 1, 0, 0],
        [0, x, y, 0, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    return V

def least_squares_method(points):
    """最小2乗法による楕円推定"""
    N = len(points)
    M = np.zeros((6, 6))
    
    for i in range(N):
        xi = xi_vector(points[i, 0], points[i, 1])
        M += np.outer(xi, xi)
    
    # 最小固有値に対応する固有ベクトル
    eigenvals, eigenvecs = eigh(M)
    u = eigenvecs[:, 0]  # 最小固有値に対応する固有ベクトル
    return u / np.linalg.norm(u)

def maximum_likelihood_estimation(points, sigma, max_iter=100, tol=1e-6):
    """最尤推定法による楕円推定"""
    N = len(points)
    
    # 初期値として最小2乗法の結果を使用
    u = least_squares_method(points)
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # 行列MとLの計算
        M = np.zeros((6, 6))
        L = np.zeros((6, 6))
        
        for i in range(N):
            xi = xi_vector(points[i, 0], points[i, 1])
            V = compute_covariance_matrix(points[i, 0], points[i, 1], sigma)
            
            denominator = np.dot(u, np.dot(V, u))
            if denominator <= 0:
                denominator = 1e-10
            
            M += np.outer(xi, xi) / denominator
            
            numerator = np.dot(xi, u)**2
            L += (numerator / denominator**2) * V
        
        # 一般化固有値問題を解く
        try:
            eigenvals, eigenvecs = eigh(M, L)
            # 最小の正の固有値に対応する固有ベクトルを選択
            valid_idx = eigenvals > 1e-10
            if np.any(valid_idx):
                min_idx = np.argmin(eigenvals[valid_idx])
                valid_indices = np.where(valid_idx)[0]
                u = eigenvecs[:, valid_indices[min_idx]]
            else:
                u = eigenvecs[:, 0]
        except:
            # 数値的問題が発生した場合は最小2乗法の結果を返す
            break
        
        u = u / np.linalg.norm(u)
        
        # 収束判定
        if np.linalg.norm(u - u_old) < tol:
            break
    
    return u

def compute_kcr_bound(true_u, points, sigma):
    """KCR下界の計算"""
    N = len(points)
    M = np.zeros((6, 6))
    
    for i in range(N):
        xi = xi_vector(points[i, 0], points[i, 1])
        V = compute_covariance_matrix(points[i, 0], points[i, 1], sigma)
        
        denominator = np.dot(true_u, np.dot(V, true_u))
        if denominator <= 0:
            denominator = 1e-10
        
        M += np.outer(xi, xi) / denominator
    
    # 固有値を計算
    eigenvals = eigh(M, eigvals_only=True)
    eigenvals = np.sort(eigenvals)[::-1]  # 降順にソート
    
    # 最小固有値（0に近い値）を除外して上位5個を使用
    valid_eigenvals = eigenvals[eigenvals > 1e-10]
    if len(valid_eigenvals) >= 5:
        D_KCR = np.sqrt(np.sum(1.0 / valid_eigenvals[:5]))
    else:
        # 有効な固有値が5個未満の場合
        D_KCR = np.sqrt(np.sum(1.0 / valid_eigenvals))
    
    return D_KCR

def compute_rms_error(true_u, estimated_u):
    """RMS誤差の計算"""
    # 単位ベクトルに正規化
    true_u = true_u / np.linalg.norm(true_u)
    estimated_u = estimated_u / np.linalg.norm(estimated_u)
    
    # 射影行列を使ったRMS誤差計算
    P = np.eye(6) - np.outer(true_u, true_u)
    error_vector = np.dot(P, estimated_u)
    return np.linalg.norm(error_vector)

def run_experiment():
    """実験の実行"""
    # パラメータ設定
    N = 100
    sigma_max = 3.0
    sigma_values = np.arange(0.1, sigma_max + 0.1, 0.1)
    n_trials = 1000
    
    # 真の楕円上の点を生成
    true_x, true_y = generate_ellipse_points(N)
    true_points = np.column_stack([true_x, true_y])
    
    # 真のパラメータベクトル（楕円 x^2/300^2 + y^2/200^2 = 1）
    # 一般形: x^2/90000 + y^2/40000 - 1 = 0
    # A=1/90000, B=0, C=1/40000, D=0, E=0, F=-1
    true_u = np.array([1/90000, 0, 1/40000, 0, 0, -1])
    true_u = true_u / np.linalg.norm(true_u)
    
    # 結果を保存する配列
    rms_lsm = np.zeros(len(sigma_values))
    rms_mle = np.zeros(len(sigma_values))
    kcr_bounds = np.zeros(len(sigma_values))
    
    print("実験開始...")
    
    for i, sigma in enumerate(sigma_values):
        print(f"σ = {sigma:.1f} の処理中...")
        
        rms_lsm_trials = []
        rms_mle_trials = []
        
        # KCR下界の計算
        kcr_bounds[i] = compute_kcr_bound(true_u, true_points, sigma)
        
        for trial in range(n_trials):
            # ノイズを追加
            noise_x = np.random.normal(0, sigma, N)
            noise_y = np.random.normal(0, sigma, N)
            noisy_points = true_points + np.column_stack([noise_x, noise_y])
            
            # 最小2乗法
            u_lsm = least_squares_method(noisy_points)
            rms_lsm_trials.append(compute_rms_error(true_u, u_lsm))
            
            # 最尤推定法
            u_mle = maximum_likelihood_estimation(noisy_points, sigma)
            rms_mle_trials.append(compute_rms_error(true_u, u_mle))
        
        rms_lsm[i] = np.mean(rms_lsm_trials)
        rms_mle[i] = np.mean(rms_mle_trials)
    
    return sigma_values, rms_lsm, rms_mle, kcr_bounds

def plot_results(sigma_values, rms_lsm, rms_mle, kcr_bounds):
    """結果のプロット"""
    plt.figure(figsize=(10, 8))
    
    plt.plot(sigma_values, rms_lsm, 'r-', linewidth=2, label='LSM (Least Squares Method)')
    plt.plot(sigma_values, rms_mle, 'b-', linewidth=2, label='MLE (Maximum Likelihood Estimation)')
    plt.plot(sigma_values, kcr_bounds, 'g-', linewidth=2, label='KCR Lower Bound')
    
    plt.xlabel('Standard Deviation σ', fontsize=12)
    plt.ylabel('RMS Error', fontsize=12)
    plt.title('Comparison of Ellipse Parameter Estimation Methods\n(N=100, Trials=1000)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, sigma_values[-1])
    plt.ylim(0, max(np.max(rms_lsm), np.max(rms_mle)) * 1.1)
    
    plt.tight_layout()
    plt.show()
    
    # 結果の表示
    print("\n=== 実験結果 ===")
    print("σ値\tLSM\t\tMLE\t\tKCR下界")
    print("-" * 50)
    for i, sigma in enumerate(sigma_values):
        print(f"{sigma:.1f}\t{rms_lsm[i]:.6f}\t{rms_mle[i]:.6f}\t{kcr_bounds[i]:.6f}")

if __name__ == "__main__":
    # 実験実行
    sigma_values, rms_lsm, rms_mle, kcr_bounds = run_experiment()
    
    # 結果のプロット
    plot_results(sigma_values, rms_lsm, rms_mle, kcr_bounds)