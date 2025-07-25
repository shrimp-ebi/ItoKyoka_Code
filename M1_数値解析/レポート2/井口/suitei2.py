import numpy as np
import cv2
import matplotlib.pyplot as plt

# 試行回数
loop = 100
# 初期パラメータ
theta = np.deg2rad(0)
s = 1
all_s = []
all_theta = []
all_error = []

# ガウシアンフィルタのパラメータ
sigma = 1.5
size = int(1 + 4 * sigma)
x = np.arange(size) - size // 2
y = np.arange(size) - size // 2
X, Y = np.meshgrid(x, y)
gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
gaussian_x = -gaussian * (X / sigma**2)
gaussian_y = -gaussian * (Y / sigma**2)

# 画像の読み込みと正規化
img_in = cv2.imread('wow.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
img_out = cv2.imread('out_wow.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

# 画像の大きさと中心
pixel = img_in.shape[0]
cx_in = cy_in = pixel / 2
cx_out = cy_out = pixel / 2

# 円内のピクセル座標を取得
Y, X = np.indices(img_in.shape)
radius = pixel / 2
mask = (X - cx_in)**2 + (Y - cy_in)**2 <= radius**2
X = X[mask]
Y = Y[mask]

# 初期画像と勾配画像
im = np.zeros(img_in.shape, dtype=float)
Ix = np.zeros(img_in.shape, dtype=float)
Iy = np.zeros(img_in.shape, dtype=float)

for i in range(loop):
    J_theta = 0
    J_2theta = 0
    J_s = 0
    J_2s = 0
    J_theta_s = 0
    error = 0

    # 現パラメータのθ分だけ回転させる
    x_rot = np.round((X - cx_in) * np.cos(-theta) - (Y - cy_in) * np.sin(-theta) + cx_in).astype(int)
    y_rot = np.round((X - cx_in) * np.sin(-theta) + (Y - cy_in) * np.cos(-theta) + cy_in).astype(int)
    
    valid_mask = (x_rot >= 0) & (x_rot < pixel) & (y_rot >= 0) & (y_rot < pixel)
    im[Y[valid_mask], X[valid_mask]] = img_out[y_rot[valid_mask], x_rot[valid_mask]]

    # 平滑微分計算
    Ixt = cv2.filter2D(im, -1, gaussian_x)
    Iyt = cv2.filter2D(im, -1, gaussian_y)

    # 画像を元の座標系に戻す
    x_rot = np.round((X - cx_in) * np.cos(theta) - (Y - cy_in) * np.sin(theta) + cx_in).astype(int)
    y_rot = np.round((X - cx_in) * np.sin(theta) + (Y - cy_in) * np.cos(theta) + cy_in).astype(int)
    
    valid_mask = (x_rot >= 0) & (x_rot < pixel) & (y_rot >= 0) & (y_rot < pixel)
    Ix[Y[valid_mask], X[valid_mask]] = Ixt[y_rot[valid_mask], x_rot[valid_mask]]
    Iy[Y[valid_mask], X[valid_mask]] = Iyt[y_rot[valid_mask], x_rot[valid_mask]]

    # 結果の画像を保存
    plt.figure()
    plt.gray()
    plt.subplot(121)
    plt.imshow(Ix, cmap='gray')
    plt.subplot(122)
    plt.imshow(Iy, cmap='gray')
    plt.savefig(f"Ix_Iy_{i}.png")
    plt.close()

    # 更新計算
    dif_x = np.round(s * ((X - cx_in) * np.cos(theta) - (cy_in - Y) * np.sin(theta)) + cx_in).astype(int)
    dif_y = np.round(-s * ((X - cx_in) * np.sin(theta) + (cy_in - Y) * np.cos(theta)) + cy_in).astype(int)

    valid_mask = (dif_x >= 0) & (dif_x < pixel) & (dif_y >= 0) & (dif_y < pixel)
    dif_I = img_out[dif_y[valid_mask], dif_x[valid_mask]]
    dif_Ix = Ix[dif_y[valid_mask], dif_x[valid_mask]]
    dif_Iy = Iy[dif_y[valid_mask], dif_x[valid_mask]]

    I = img_in[Y[valid_mask], X[valid_mask]]
    dx_theta = s * (-(X[valid_mask] - cx_in) * np.sin(theta) - (cy_in - Y[valid_mask]) * np.cos(theta))
    dy_theta = s * ((X[valid_mask] - cx_in) * np.cos(theta) - (cy_in - Y[valid_mask]) * np.sin(theta))
    dx_s = (X[valid_mask] - cx_in) * np.cos(theta) - (cy_in - Y[valid_mask]) * np.sin(theta)
    dy_s = (X[valid_mask] - cx_in) * np.sin(theta) + (cy_in - Y[valid_mask]) * np.cos(theta)
    dx_theta_s = (-(X[valid_mask] - cx_in) * np.sin(theta) - (cy_in - Y[valid_mask]) * np.cos(theta))
    dy_theta_s = ((X[valid_mask] - cx_in) * np.cos(theta) - (cy_in - Y[valid_mask]) * np.sin(theta))

    J_theta += np.sum((dif_I - I) * (dif_Ix * dx_theta + dif_Iy * dy_theta))
    J_2theta += np.sum((dif_Ix * dx_theta + dif_Iy * dy_theta) ** 2)
    J_s += np.sum((dif_I - I) * (dif_Ix * dx_s + dif_Iy * dy_s))
    J_2s += np.sum((dif_Ix * dx_s + dif_Iy * dy_s) ** 2)
    J_theta_s += np.sum(dif_Ix**2 * dx_theta * dx_s + dif_Ix * dif_Ix * (dx_theta * dy_s + dx_s * dy_theta) + dif_Iy**2 * dy_theta * dy_s)

    error += 0.5 * np.sum((dif_I - I) ** 2)

    # 現在の結果を保存
    plt.figure()
    plt.gray()
    plt.subplot(121)
    plt.imshow(img_in, cmap='gray')
    plt.subplot(122)
    plt.imshow(im, cmap='gray')
    plt.savefig(f"img_in_vs_im_{i}.png")
    plt.close()

    J = np.array([[J_2theta, J_theta_s], [J_theta_s, J_2s]])
    J_vec = np.array([J_theta, J_s])

    # ガウス・ニュートン法でパラメータを更新
    dtheta, ds = np.linalg.solve(J, J_vec)

    theta -= dtheta
    s -= ds

    all_s.append(s)
    all_theta.append(np.rad2deg(theta))
    all_error.append(error)

print("Estimated Theta (degrees):", np.rad2deg(theta))
print("Estimated Scale:", s)

np.savetxt("s_estimates.csv", all_s, delimiter=",", fmt='%f')
np.savetxt("theta_estimates.csv", all_theta, delimiter=",", fmt='%f')
