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
import cv2
import numpy as np
import matplotlib
# Use non-interactive backend to avoid Qt errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def apply_smoothing_differential_filter(img, kernel_size=5, sigma=2):
    """
    Compute smoothed image gradients using a simple difference filter on a blurred image.
    Returns dx, dy as float64 arrays.
    """
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (kernel_size, kernel_size), sigmaX=sigma).astype(np.float64)
    kernel_dx = np.array([[-1, 0, 1]], dtype=np.float64)
    kernel_dy = kernel_dx.T
    dx = cv2.filter2D(blurred, cv2.CV_64F, kernel_dx)
    dy = cv2.filter2D(blurred, cv2.CV_64F, kernel_dy)
    return dx, dy

def estimate_by_gauss_newton(img_input, img_transformed, theta_init=0.0, scale_init=1.0, threshold=1e-5, max_iter=100):
    theta = np.deg2rad(theta_init)
    scale = scale_init
    H, W = img_input.shape[:2]
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_coords = x_coords - W/2
    y_coords = y_coords - H/2
    radius = min(W, H) / 2
    mask = (x_coords**2 + y_coords**2) <= radius**2

    I = img_transformed.astype(np.float64)
    I_prime_org = img_input
    theta_history = []
    scale_history = []

    for _ in range(max_iter):
        a = scale * np.cos(theta)
        b = scale * np.sin(theta)
        M = np.array([[a, -b, 0], [b, a, 0], [0, 0, 1]], dtype=np.float64)
        M_inv = np.linalg.inv(M)
        pts = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())])
        uv = M_inv @ pts
        x_org = uv[0] + W/2
        y_org = uv[1] + H/2
        x_i = np.round(x_org).astype(int)
        y_i = np.round(y_org).astype(int)
        I_prime = np.zeros_like(I)
        valid = (x_i>=0)&(x_i<W)&(y_i>=0)&(y_i<H)
        I_prime_flat = I_prime.flatten()
        I_prime_flat[valid] = I_prime_org[y_i[valid], x_i[valid]]
        I_prime = I_prime_flat.reshape(H, W)
        I_dx, I_dy = apply_smoothing_differential_filter(I_prime)
        diff = I_prime - I
        dx_dtheta = -scale * (x_coords*np.sin(theta) + y_coords*np.cos(theta))
        dy_dtheta = scale*(x_coords*np.cos(theta) - y_coords*np.sin(theta))
        dx_dscale = x_coords*np.cos(theta) - y_coords*np.sin(theta)
        dy_dscale = x_coords*np.sin(theta) + y_coords*np.cos(theta)
        dtheta_term = (I_dx*dx_dtheta + I_dy*dy_dtheta)[mask]
        dscale_term = (I_dx*dx_dscale + I_dy*dy_dscale)[mask]
        diff_mask = diff[mask]
        J_theta = np.sum(diff_mask * dtheta_term)
        J_scale = np.sum(diff_mask * dscale_term)
        J_tt = np.sum(dtheta_term**2)
        J_ss = np.sum(dscale_term**2)
        J_ts = np.sum(dtheta_term * dscale_term)
        nabla = np.array([J_theta, J_scale])
        H_approx = np.array([[J_tt, J_ts], [J_ts, J_ss]])
        try:
            delta = np.linalg.solve(H_approx, nabla)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Stopping.")
            break
        theta -= delta[0]
        scale -= delta[1]
        theta_history.append(theta)
        scale_history.append(scale)
        if np.linalg.norm(delta) < threshold:
            break

    return theta, scale, theta_history, scale_history

def main():
    if len(sys.argv) < 3:
        print("Usage: python gauss.py input_image transformed_image [theta_init_deg] [scale_init]")
        sys.exit(1)
    input_path = sys.argv[1]
    transformed_path = sys.argv[2]
    theta_init = float(sys.argv[3]) if len(sys.argv)>=4 else 0.0
    scale_init = float(sys.argv[4]) if len(sys.argv)>=5 else 1.0
    img_input = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img_transformed = cv2.imread(transformed_path, cv2.IMREAD_GRAYSCALE)
    if img_input is None or img_transformed is None:
        print("Error: cannot open input images.")
        sys.exit(1)
    theta, scale, th_hist, sc_hist = estimate_by_gauss_newton(img_input, img_transformed, theta_init, scale_init)
    print(f"Estimated rotation (deg): {np.rad2deg(theta):.4f}")
    print(f"Estimated scale: {scale:.4f}")
    # save convergence plot
    plt.figure()
    plt.plot([np.rad2deg(t) for t in th_hist], label='theta (deg)')
    plt.plot(sc_hist, label='scale')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence.png")
    print("Convergence plot saved to convergence.png")

if __name__ == "__main__":
    main()
        
        