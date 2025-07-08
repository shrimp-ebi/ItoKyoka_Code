# 最小2乗法・最尤推定法・KCR下界に基づくRMS誤差評価プログラム

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.linalg import eigh

# --- データ生成関連関数 ---
def generate_ellipse_points(N=100):
    theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)
    x = 300 * np.cos(theta)
    y = 200 * np.sin(theta)
    return x, y

def add_noise(x, y, sigma):
    x_noise = np.random.normal(0, sigma, len(x))
    y_noise = np.random.normal(0, sigma, len(y))
    return x + x_noise, y + y_noise

# --- 楕円パラメータ推定 ---
def least_squares_ellipse(x, y):
    xi = np.column_stack([x**2, 2*x*y, y**2, 2*x, 2*y, np.ones(len(x))])
    M = xi.T @ xi
    _, eigenvectors = eigh(M)
    u = eigenvectors[:, 0]
    return u

def covariance_matrix_xi(x, y, sigma):
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
    u_true = u_true / np.linalg.norm(u_true)
    rms_total = 0
    for u in u_estimates:
        u = u / np.linalg.norm(u)
        if np.dot(u, u_true) < 0:
            u = -u
        delta_u = (np.eye(6) - np.outer(u_true, u_true)) @ u
        rms_total += np.linalg.norm(delta_u)**2
    return np.sqrt(rms_total / len(u_estimates))

# --- KCR下界計算 ---
def calculate_kcr_lower_bound(x, y, u_true, sigma):
    N = len(x)
    M = np.zeros((6, 6))
    for i in range(N):
        xi = np.array([x[i]**2, 2*x[i]*y[i], y[i]**2, 2*x[i], 2*y[i], 1])
        V_xi = covariance_matrix_xi(x[i], y[i], sigma)
        uVu = u_true.T @ V_xi @ u_true
        if uVu > 1e-12:
            M += np.outer(xi, xi) / uVu
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.sort(eigvals)[::-1]  # 大きい順にソート
    eigvals = eigvals[:5]  # 自由度5
    if np.any(eigvals <= 0):
        return np.nan
    return np.sqrt(np.sum(1 / eigvals))

# --- メイン処理 ---
def main():
    N = 100
    sigma_values = np.arange(0.1, 3.1, 0.1)
    num_trials = 1000
    u_true = np.array([1/300**2, 0, 1/200**2, 0, 0, -1])
    u_true /= np.linalg.norm(u_true)

    rms_ls = []
    rms_mle = []
    kcr_bounds = []

    for idx, sigma in enumerate(sigma_values):
        print(f"\n[\u03c3 = {sigma:.1f}] {idx+1}/{len(sigma_values)} 計算中...")
        u_ls_list = []
        u_mle_list = []
        x_true, y_true = generate_ellipse_points(N)
        for trial in range(num_trials):
            if trial % 200 == 0:
                print(f"  trial {trial}/{num_trials}...")
            x_noisy, y_noisy = add_noise(x_true, y_true, sigma)
            u_ls = least_squares_ellipse(x_noisy, y_noisy)
            u_mle = maximum_likelihood_ellipse(x_noisy, y_noisy, sigma)
            u_ls_list.append(u_ls)
            u_mle_list.append(u_mle)
        rms_ls.append(calculate_rms_error(u_ls_list, u_true))
        rms_mle.append(calculate_rms_error(u_mle_list, u_true))
        kcr_bound = calculate_kcr_lower_bound(x_true, y_true, u_true, sigma)
        kcr_bounds.append(kcr_bound)
        print(f"  LSM RMS = {rms_ls[-1]:.4e}, MLE RMS = {rms_mle[-1]:.4e}, KCR = {kcr_bound:.4e}")

    # --- グラフ描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_values, rms_ls, 'r-o', label='LSM')
    plt.plot(sigma_values, rms_mle, 'b-s', label='MLE')
    plt.plot(sigma_values, kcr_bounds, 'g-^', label='KCR Lower Bound')
    plt.xlabel('Noise Level $\sigma$')
    plt.ylabel('RMS Error')
    plt.grid(True)
    plt.legend()
    plt.title('RMS Error and KCR Bound vs. Noise Level')
    plt.savefig('5-2.png', dpi=300)
    print("図を保存しました：5-2.png")
    plt.show()

if __name__ == '__main__':
    main()