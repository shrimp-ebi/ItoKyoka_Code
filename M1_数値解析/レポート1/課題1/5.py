import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.linalg import eigh, solve
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
    """ξ空間での共分散行列を計算"""
    V = np.zeros((6, 6))
    V[0, 0] = x**2
    V[0, 1] = V[1, 0] = x * y
    V[0, 3] = V[3, 0] = x
    V[1, 1] = x**2 + y**2
    V[1, 2] = V[2, 1] = x * y
    V[1, 3] = V[3, 1] = y
    V[1, 4] = V[4, 1] = x
    V[2, 2] = y**2
    V[2, 4] = V[4, 2] = y
    V[3, 3] = 1
    V[4, 4] = 1
    return 4 * sigma**2 * V

def maximum_likelihood_ellipse(x, y, sigma, max_iter=50, tol=1e-6):
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
        if np.linalg.norm(u - u_old) < tol:
            break
    
    return u

def ellipse_parameters_from_u(u):
    """パラメータベクトルuから楕円パラメータを抽出"""
    A, B, C, D, E, F = u
    
    # 楕円の中心
    det = B**2 - A*C
    if abs(det) < 1e-12:
        return None
    
    cx = (C*D - B*E) / det
    cy = (A*E - B*D) / det
    
    # 長軸・短軸の長さ
    num = 2*(A*E**2 + C*D**2 + F*B**2 - 2*B*D*E - A*C*F)
    
    temp1 = A + C
    temp2 = np.sqrt((A - C)**2 + 4*B**2)
    
    a = np.sqrt(num / (det * (temp2 - temp1)))  # 長軸
    b = np.sqrt(num / (det * (-temp2 - temp1))) # 短軸
    
    # 回転角
    if abs(B) < 1e-12:
        if A < C:
            theta = 0
        else:
            theta = np.pi/2
    else:
        theta = 0.5 * np.arctan(2*B / (A - C))
    
    return cx, cy, a, b, theta

def calculate_rms_error(u_estimated, u_true):
    """RMS誤差を計算"""
    # 真値ベクトルを正規化
    u_true_norm = u_true / np.linalg.norm(u_true)
    u_est_norm = u_estimated / np.linalg.norm(u_estimated)
    
    # 符号の調整（内積が負の場合は符号を反転）
    if np.dot(u_true_norm, u_est_norm) < 0:
        u_est_norm = -u_est_norm
    
    # 射影行列 P_u = I - u_true * u_true^T
    P_u = np.eye(6) - np.outer(u_true_norm, u_true_norm)
    
    # Δu = P_u * u_estimated
    delta_u = P_u @ u_est_norm
    
    # RMS誤差
    rms = np.linalg.norm(delta_u)
    
    return rms

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
    eigenvalues = np.linalg.eigvals(M)
    eigenvalues = np.sort(eigenvalues)[::-1]  # 降順にソート
    
    # モデルの自由度r=5（楕円は6パラメータ-1制約=5自由度）
    r = 5
    
    # KCR下界を計算
    D_KCR = 0
    for k in range(r):
        if eigenvalues[k] > 1e-12:  # 数値的安定性のチェック
            D_KCR += 1.0 / eigenvalues[k]
    
    return np.sqrt(D_KCR)

def main():
    # パラメータ設定
    N = 100
    sigma_max = 2.0
    sigma_values = np.arange(0.1, sigma_max + 0.1, 0.1)
    num_trials = 1000
    
    u_true = np.array([1/300**2, 0, 1/200**2, 0, 0, -1])
    
    # 結果を格納する配列
    rms_least_squares = np.zeros(len(sigma_values))
    rms_maximum_likelihood = np.zeros(len(sigma_values))
    kcr_bounds = np.zeros(len(sigma_values))  # KCR下界用を追加
    
    print("Starting ellipse fitting experiment...")
    print(f"Noise level: {sigma_values[0]:.1f} to {sigma_values[-1]:.1f}")
    print(f"Number of trials: {num_trials}")
    
    for i, sigma in enumerate(sigma_values):
        print(f"σ = {sigma:.1f} の処理中... ({i+1}/{len(sigma_values)})")
        
        rms_ls_trials = []
        rms_ml_trials = []
        
        # KCR下界を計算（真値の楕円点を使用）
        x_true, y_true = generate_ellipse_points(N)
        kcr_bounds[i] = calculate_kcr_bound(x_true, y_true, sigma, u_true)
        
        for trial in range(num_trials):
            # ノイズ付きデータの生成
            x_true_trial, y_true_trial = generate_ellipse_points(N)
            x_noisy, y_noisy = add_noise(x_true_trial, y_true_trial, sigma)
            
            try:
                # 最小2乗法
                u_ls = least_squares_ellipse(x_noisy, y_noisy)
                rms_ls = calculate_rms_error(u_ls, u_true)
                rms_ls_trials.append(rms_ls)
                
                # 最尤推定法
                u_ml = maximum_likelihood_ellipse(x_noisy, y_noisy, sigma)
                rms_ml = calculate_rms_error(u_ml, u_true)
                rms_ml_trials.append(rms_ml)
                
            except:
                continue
        
        # 平均RMS誤差を計算
        if rms_ls_trials:
            rms_least_squares[i] = np.mean(rms_ls_trials)
        if rms_ml_trials:
            rms_maximum_likelihood[i] = np.mean(rms_ml_trials)
    
    # 結果のプロット
    plt.figure(figsize=(12, 8))
    plt.plot(sigma_values, rms_least_squares, 'r-o', label='LSM', markersize=4)
    plt.plot(sigma_values, rms_maximum_likelihood, 'b-s', label='MLE', markersize=4)
    plt.plot(sigma_values, kcr_bounds, 'g-^', label='KCR', markersize=4)  # KCR下界を追加

    plt.xlabel('Noise Level σ')
    plt.ylabel('RMS Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, sigma_max)
    
    # y軸の上限を調整（KCR下界も考慮）
    max_val = max(np.max(rms_least_squares), np.max(rms_maximum_likelihood), np.max(kcr_bounds))
    plt.ylim(0, max_val * 1.1)
    
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.savefig('5.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
if __name__ == "__main__":
    main()