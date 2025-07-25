import numpy as np
import cv2
from math import cos, sin, radians
import matplotlib
matplotlib.use("Agg")  # GUIを使わずに画像描画する非インタラクティブモード

import matplotlib.pyplot as plt

def transform_coords(x, y, theta, s):
    x_prime = s * (x * np.cos(theta) - y * np.sin(theta))
    y_prime = s * (x * np.sin(theta) + y * np.cos(theta))
    return x_prime, y_prime

def compute_gradient(img):
    # ガウシアンフィルタとSobel微分で平滑微分画像を生成
    img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=2)
    grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

def interpolate(img, x, y):
    # バイリニア補間
    h, w = img.shape
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return 0.0
    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0
    val = (1 - dx) * (1 - dy) * img[y0, x0] + \
          dx * (1 - dy) * img[y0, x0+1] + \
          (1 - dx) * dy * img[y0+1, x0] + \
          dx * dy * img[y0+1, x0+1]
    return val

def gauss_newton(I, I_prime, theta_init, s_init, max_iter=50, eps=1e-4):
    H, W = I.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    x = X - W // 2
    y = Y - H // 2

    Ix, Iy = compute_gradient(I_prime)
    theta, s = theta_init, s_init

    history = []

    for it in range(max_iter):
        J_theta = 0.0
        J_s = 0.0
        J_tt = 0.0
        J_ss = 0.0
        J_ts = 0.0

        for i in range(H):
            for j in range(W):
                xi = x[i, j]
                yi = y[i, j]
                xp, yp = transform_coords(xi, yi, theta, s)
                xp += W // 2
                yp += H // 2

                I_val = I[i, j]
                I_p_val = interpolate(I_prime, xp, yp)
                diff = I_p_val - I_val

                Ix_p = interpolate(Ix, xp, yp)
                Iy_p = interpolate(Iy, xp, yp)

                dx_dtheta = -s * (xi * np.sin(theta) + yi * np.cos(theta))
                dy_dtheta =  s * (xi * np.cos(theta) - yi * np.sin(theta))
                dx_ds = xi * np.cos(theta) - yi * np.sin(theta)
                dy_ds = xi * np.sin(theta) + yi * np.cos(theta)

                g_theta = Ix_p * dx_dtheta + Iy_p * dy_dtheta
                g_s     = Ix_p * dx_ds + Iy_p * dy_ds

                J_theta += diff * g_theta
                J_s += diff * g_s
                J_tt += g_theta ** 2
                J_ss += g_s ** 2
                J_ts += g_theta * g_s

        # ヘッセ行列と勾配ベクトルの構築
        H_mat = np.array([[J_tt, J_ts], [J_ts, J_ss]])
        g_vec = np.array([J_theta, J_s])
        try:
            delta = np.linalg.solve(H_mat, g_vec)
        except np.linalg.LinAlgError:
            print("ヘッセ行列が特異です")
            break

        theta -= delta[0]
        s -= delta[1]
        history.append((theta, s))

        if np.linalg.norm(delta) < eps:
            break

    return theta, s, history

if __name__ == "__main__":
    I = cv2.imread("shrimp.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    I_prime = cv2.imread("transformed.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

    theta_init = np.deg2rad(0)
    s_init = 1.0

    theta, s, hist = gauss_newton(I, I_prime, theta_init, s_init)

    print(f"推定された回転角度 θ: {np.rad2deg(theta):.4f}°")
    print(f"推定されたスケール s: {s:.4f}")

    theta_vals = [np.rad2deg(t) for t, _ in hist]
    s_vals = [s_ for _, s_ in hist]

    plt.plot(theta_vals, label='θ (degree)')
    plt.plot(s_vals, label='s')
    plt.xlabel("Iteration")
    plt.legend()
    plt.title("収束過程")
    plt.grid()
    plt.savefig("convergence_plot.png")
    print("収束過程のグラフを 'convergence_plot.png' に保存しました。")
