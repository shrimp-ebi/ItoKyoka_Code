import numpy as np
import matplotlib.pyplot as plt

# 点の数
N = 100

# θ_i の計算
theta = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)

# 楕円上の点列 (x_i, y_i)
x = 300 * np.cos(theta)
y = 200 * np.sin(theta)

# 描画
plt.figure(figsize=(10, 10))
plt.plot(x, y, 'bo', label='Points on Ellipse')  # 青丸点でプロット
plt.gca().set_aspect('equal')  # アスペクト比を1:1に設定
plt.grid(True)
plt.title("Point sequence distribution on ellipse")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# 点の終端が軸にくっつくように軸の範囲を設定
x_margin = (x.max() - x.min()) * 0.02
y_margin = (y.max() - y.min()) * 0.02
plt.xlim(x.min() - x_margin, x.max() + x_margin)
plt.ylim(y.min() - y_margin, y.max() + y_margin)

# 図を保存
plt.savefig(f'ellipse_points_N{N}.png', dpi=300, bbox_inches='tight')
plt.show()

# 結果の確認
print(f"生成された点の数: {N}")
print(f"θの範囲: {theta[0]:.4f} ～ {theta[-1]:.4f}")
print(f"θの範囲（度）: {np.degrees(theta[0]):.2f}° ～ {np.degrees(theta[-1]):.2f}°")
print(f"x座標の範囲: {x.min():.2f} ～ {x.max():.2f}")
print(f"y座標の範囲: {y.min():.2f} ～ {y.max():.2f}")