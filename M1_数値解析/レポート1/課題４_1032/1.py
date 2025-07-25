import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

np.random.seed(42)

# === 楕円とノイズ生成 ===
def generate_ellipse_points(N):
    theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)
    x = 300 * np.cos(theta)
    y = 200 * np.sin(theta)
    return x, y

def add_noise(x, y, sigma):
    x_noisy = x + np.random.normal(0, sigma, size=x.shape)
    y_noisy = y + np.random.normal(0, sigma, size=y.shape)
    return x_noisy, y_noisy

# === 真のパラメータ ===
u_true = np.array([1.111e-5, 0, 2.5e-5, 0, 0, -1.0])  # A, B, C, D, E, F

# === デザイン行列構築 ===
def build_design_matrix(x, y):
    return np.vstack([
        x**2, x*y, y**2, x, y, np.ones_like(x)
    ]).T

# === 最小二乗法による楕円推定 ===
def fit_ellipse_lsm(x, y):
    D = build_design_matrix(x, y)
    _, _, Vt = np.linalg.svd(D)
    u = Vt[-1]
    return normalize_param_vector(u)

# === 最尤推定法（ラグランジュ） ===
def fit_ellipse_mle(x, y):
    D = build_design_matrix(x, y)
    N = len(x)
    V = np.identity(6) * 9.0  # 各点の共分散行列をスカラー倍で仮定
    M = np.zeros((6, 6))
    for i in range(N):
        xi = D[i][:, None]  # (6, 1)
        M += (xi @ xi.T) / (u_true @ V @ u_true)
    eigvals, eigvecs = eigh(M)
    u = eigvecs[:, -2]  # 最小ではなく2番目（0を除いた最大の固有値）
    return normalize_param_vector(u)

# === パラメータ正規化（最小二乗と同じスケールに揃える）===
def normalize_param_vector(u):
    return u / np.linalg.norm(u)

# === RMS誤差評価 ===
def calc_rms_error(U, u_true):
    diffs = U - u_true[None, :]
    return np.sqrt(np.mean(np.sum(diffs**2, axis=1)))

# === KCR下界評価 ===
def calc_kcr_bound(x, y, u_true):
    D = build_design_matrix(x, y)
    N = len(x)
    V = np.identity(6) * 9.0
    M = np.zeros((6, 6))
    for i in range(N):
        xi = D[i][:, None]
        M += (xi @ xi.T) / (u_true @ V @ u_true)
    eigvals = eigh(M, eigvals_only=True)[::-1]  # 降順
    return np.sqrt(np.sum(1 / eigvals[:5]))  # λ6=0 を除く


# シミュレーション設定
N = 100
sigma_list = np.arange(0.1, 3.0 + 0.1, 0.1)  # 0.1 ~ 3.0
n_trials = 1000

rms_lsm_list = []
rms_mle_list = []
kcr_list = []

# 真の楕円点列（共通）
x_true, y_true = generate_ellipse_points(N)
D_true = build_design_matrix(x_true, y_true)

print("Simulating...")

for sigma in sigma_list:
    lsm_params = []
    mle_params = []

    for _ in range(n_trials):
        # ノイズを加える
        x_noisy, y_noisy = add_noise(x_true, y_true, sigma)

        # LSM推定
        try:
            u_lsm = fit_ellipse_lsm(x_noisy, y_noisy)
            lsm_params.append(u_lsm)
        except:
            continue  # エラー時スキップ

        # MLE推定
        try:
            u_mle = fit_ellipse_mle(x_noisy, y_noisy)
            mle_params.append(u_mle)
        except:
            continue  # エラー時スキップ

    # RMS誤差の計算
    lsm_params = np.array(lsm_params)
    mle_params = np.array(mle_params)

    rms_lsm = calc_rms_error(lsm_params, normalize_param_vector(u_true))
    rms_mle = calc_rms_error(mle_params, normalize_param_vector(u_true))

    # KCR下界の計算（ノイズなしの真値点列を使用）
    D_u = normalize_param_vector(u_true)
    kcr = calc_kcr_bound(x_true, y_true, D_u)

    # 結果を保存
    rms_lsm_list.append(rms_lsm)
    rms_mle_list.append(rms_mle)
    kcr_list.append(kcr)

    print(f"σ={sigma:.1f}: RMS_LSM={rms_lsm:.4f}, RMS_MLE={rms_mle:.4f}, KCR={kcr:.4f}")


# グラフ描画
plt.figure(figsize=(10, 6))
plt.plot(sigma_list, rms_lsm_list, 'o-', label="Least Squares (LSM)", color='blue')
plt.plot(sigma_list, rms_mle_list, 's-', label="Maximum Likelihood (MLE)", color='red')
plt.plot(sigma_list, kcr_list, 'k--', label="KCR Bound", linewidth=2)

plt.title("RMS Error vs σ (N=100, 1000 trials per σ)")
plt.xlabel("σ (Standard Deviation of Noise)")
plt.ylabel("RMS Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rms_vs_sigma.png", dpi=300)
# 結果の保存
plt.savefig("課題4.png", dpi=300, bbox_inches='tight')
plt.show()

