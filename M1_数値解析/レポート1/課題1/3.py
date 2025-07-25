import numpy as np
import matplotlib.pyplot as plt

# 点の数
N = 100

# θ_i の計算
theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)

# 楕円上の点列 (x_i, y_i)
x = 300 * np.cos(theta)
y = 200 * np.sin(theta)

# 正規分布に従う誤差を生成
np.random.seed(42)  # 再現性のために固定シード
mu = 0      # 平均
sigma = 3.0 # 標準偏差

# x座標とy座標に独立に誤差を加える
x_noise = np.random.normal(mu, sigma, N)
y_noise = np.random.normal(mu, sigma, N)

# 誤差を加えた点列
x_noisy = x + x_noise
y_noisy = y + y_noise

# 描画
plt.figure(figsize=(10, 10))

# 元の点列（課題2の結果）
plt.plot(x, y, 'bo', label='Points on Ellipse')

# 誤差を加えた点列
plt.plot(x_noisy, y_noisy, 'ro', markersize=4, alpha=0.7, label='Noisy Points (σ=3.0)')

plt.gca().set_aspect('equal')  # アスペクト比を1:1に設定
plt.grid(True)
plt.title("Ellipse Points with Gaussian Noise")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# 軸の範囲を調整
all_x = np.concatenate([x, x_noisy])
all_y = np.concatenate([y, y_noisy])
x_margin = (all_x.max() - all_x.min()) * 0.02
y_margin = (all_y.max() - all_y.min()) * 0.02
plt.xlim(all_x.min() - x_margin, all_x.max() + x_margin)
plt.ylim(all_y.min() - y_margin, all_y.max() + y_margin)

# 図を保存
plt.savefig(f'ellipse_points_with_noise_N{N}.png', dpi=300, bbox_inches='tight')
plt.show()

# 結果の確認
print(f"生成された点の数: {N}")
print(f"誤差の統計情報:")
print(f"  x方向誤差 - 平均: {x_noise.mean():.3f}, 標準偏差: {x_noise.std():.3f}")
print(f"  y方向誤差 - 平均: {y_noise.mean():.3f}, 標準偏差: {y_noise.std():.3f}")
print(f"誤差を加えた後の座標範囲:")
print(f"  x座標: {x_noisy.min():.2f} ～ {x_noisy.max():.2f}")
print(f"  y座標: {y_noisy.min():.2f} ～ {y_noisy.max():.2f}")