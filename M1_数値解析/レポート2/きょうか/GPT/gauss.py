"""
元画像と変換後画像からガウス・ニュートン法で回転角度とスケールを推定し、
推定値と収束の様子をプロットします。

コマンドライン引数で入力画像と変形後の画像を指定し、
初期の回転角度とスケールをオプションで設定可能。
収束過程をプロットし、最終的な推定値を表示します。

実行方法
python gauss.py input_image transformed_image [theta_init_deg] [scale_init] 
- input_image: 入力画像のパス
- transformed_image: 変形後の画像のパス
- theta_init_deg: 初期回転角度（度単位、デフォルトは0）
- scale_init: 初期スケール（デフォルトは1.0）


"""

import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = int(3 * sigma)
    x = np.arange(-radius, radius+1)
    ker = np.exp(-x**2/(2*sigma**2))
    return ker/ker.sum()


def apply_gaussian_blur(img, sigma):
    k = gaussian_kernel1d(sigma)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=1, arr=img)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=0, arr=tmp)


def compute_image_gradients(img, sigma=2.0):
    blurred = apply_gaussian_blur(img.astype(np.float64), sigma)
    dy, dx = np.gradient(blurred)
    return dx, dy


def estimate_by_gauss_newton(img_i, img_t, theta0=0.0, scale0=1.0,
                              thresh=1e-5, max_iter=100):
    theta = np.deg2rad(theta0)
    scale = scale0
    H, W = img_i.shape
    ys, xs = np.indices((H, W))
    x = xs - (W - 1)/2.0
    y = ys - (H - 1)/2.0
    mask = x**2 + y**2 <= (min(W, H)/2.0)**2

    th_hist, sc_hist = [], []
    grad_th, grad_sc = [], []
    cost_hist, delta_norm = [], []

    for _ in range(max_iter):
        a, b = scale*np.cos(theta), scale*np.sin(theta)
        src_x = a*x + b*y + (W - 1)/2.0
        src_y = -b*x + a*y + (H - 1)/2.0
        xi = np.clip(np.round(src_x).astype(int), 0, W-1)
        yi = np.clip(np.round(src_y).astype(int), 0, H-1)
        I_p = img_i[yi, xi]
        I_dx, I_dy = compute_image_gradients(I_p)
        diff = (I_p - img_t)[mask]
        dxdth = -scale*(x*np.sin(theta) + y*np.cos(theta))
        dydth =  scale*(x*np.cos(theta) - y*np.sin(theta))
        dxdsc = x*np.cos(theta) - y*np.sin(theta)
        dydsc = x*np.sin(theta) + y*np.cos(theta)
        A = (I_dx*dxdth + I_dy*dydth)[mask]
        B = (I_dx*dxdsc + I_dy*dydsc)[mask]
        Jt = np.sum(diff*A)
        Js = np.sum(diff*B)
        Htt, Hss = np.sum(A*A), np.sum(B*B)
        Hts = np.sum(A*B)
        H_approx = np.array([[Htt, Hts],[Hts, Hss]])
        g = np.array([Jt, Js])

        cost_hist.append(0.5*np.sum(diff**2))
        th_hist.append(theta)
        sc_hist.append(scale)
        grad_th.append(Jt)
        grad_sc.append(Js)

        try:
            delta = np.linalg.solve(H_approx, g)
        except np.linalg.LinAlgError:
            print("Singular Hessian; stopping.")
            break
        theta -= delta[0]
        scale -= delta[1]
        delta_norm.append(np.linalg.norm(delta))
        if np.linalg.norm(delta) < thresh:
            break

    return theta, scale, th_hist, sc_hist, grad_th, grad_sc, cost_hist, delta_norm


def main():
    if len(sys.argv) < 3:
        print("Usage: python gauss.py input.png transformed.png [theta0] [scale0]")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    t0 = float(sys.argv[3]) if len(sys.argv)>3 else 0.0
    s0 = float(sys.argv[4]) if len(sys.argv)>4 else 1.0
    img_i = np.array(Image.open(inp).convert('L'))
    img_t = np.array(Image.open(outp).convert('L'))
    theta, scale, th_h, sc_h, gth, gsc, cost_h, dn_h = \
        estimate_by_gauss_newton(img_i, img_t, t0, s0)
    print(f"Estimated rotation (deg): {np.rad2deg(theta):.4f}")
    print(f"Estimated scale: {scale:.4f}")

    it = np.arange(len(th_h))
    # parameter histories
    plt.figure(); plt.plot(it, np.rad2deg(th_h), label='Theta (deg)'); plt.plot(it, sc_h, label='Scale'); plt.xlabel('Iteration'); plt.legend(); plt.grid(True); plt.savefig('param_evolution.png')
    # individual params
    plt.figure(); plt.plot(it, np.rad2deg(th_h)); plt.xlabel('Iteration'); plt.ylabel('Theta (deg)'); plt.grid(True); plt.savefig('theta_history.png')
    plt.figure(); plt.plot(it, sc_h); plt.xlabel('Iteration'); plt.ylabel('Scale'); plt.grid(True); plt.savefig('scale_history.png')
    # gradients
    plt.figure(); plt.plot(it, gth); plt.xlabel('Iteration'); plt.ylabel('J_theta'); plt.grid(True); plt.savefig('grad_theta.png')
    plt.figure(); plt.plot(it, gsc); plt.xlabel('Iteration'); plt.ylabel('J_scale'); plt.grid(True); plt.savefig('grad_scale.png')
    # cost and delta
    plt.figure(); plt.plot(it, cost_h); plt.xlabel('Iteration'); plt.ylabel('Cost'); plt.grid(True); plt.savefig('cost_history.png')
    plt.figure(); plt.plot(it, dn_h); plt.xlabel('Iteration'); plt.ylabel('Delta Norm'); plt.grid(True); plt.savefig('delta_norm.png')
    # solution trajectory
    plt.figure(); plt.plot(np.rad2deg(th_h), sc_h, marker='o'); plt.xlabel('Theta (deg)'); plt.ylabel('Scale'); plt.grid(True); plt.savefig('trajectory.png')
    print("Saved all history plots.")

if __name__ == "__main__":
    main()
