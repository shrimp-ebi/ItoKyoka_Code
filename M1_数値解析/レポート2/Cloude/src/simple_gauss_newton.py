"""
簡素化したガウス・ニュートン法による相似変換パラメータ推定
理論通りの実装で確実に動作させる
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要
import matplotlib.pyplot as plt
import sys

def create_similarity_transform_matrix(theta_deg, scale, tx=0, ty=0):
    """相似変換行列を作成"""
    theta_rad = np.deg2rad(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    M = np.array([
        [scale * cos_t, -scale * sin_t, tx],
        [scale * sin_t,  scale * cos_t, ty]
    ], dtype=np.float32)
    
    return M

def apply_transform(image, M):
    """画像に変換を適用"""
    h, w = image.shape[:2]
    transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return transformed

def crop_to_circle(image):
    """画像を円形にクロップ"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center) - 10
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = cv2.bitwise_and(image[:, :, i], mask)
    else:
        result = cv2.bitwise_and(image, mask)
    
    return result

def compute_derivatives(image, kernel_size=5, sigma=1.5):
    """平滑微分画像を計算"""
    # ガウシアン平滑化
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Sobel微分
    dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    return dx, dy

def gauss_newton_estimation(input_image, output_image, theta_init=30, scale_init=2, 
                           max_iterations=50, threshold=1e-4):
    """
    ガウス・ニュートン法による推定
    """
    # 画像リサイズ
    h, w = input_image.shape[:2]
    if max(h, w) > 256:
        scale_factor = 256 / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        input_image = cv2.resize(input_image, (new_w, new_h))
        output_image = cv2.resize(output_image, (new_w, new_h))
        print(f"画像リサイズ: {w}x{h} → {new_w}x{new_h}")
    
    # 正規化
    I_input = input_image.astype(np.float64) / 255.0
    I_output = output_image.astype(np.float64) / 255.0
    
    h, w = I_input.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # 円形マスク
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_rel = x_coords - center_x
    y_rel = y_coords - center_y
    radius = min(center_x, center_y) - 5
    mask = (x_rel**2 + y_rel**2) <= radius**2
    
    # 初期パラメータ
    theta = np.deg2rad(theta_init)
    scale = scale_init
    
    # 履歴
    theta_history = [np.rad2deg(theta)]
    scale_history = [scale]
    objective_history = []
    
    print(f"初期値: θ = {np.rad2deg(theta):.2f}°, s = {scale:.4f}")
    print(f"有効ピクセル数: {np.sum(mask)}")
    
    for iteration in range(max_iterations):
        # 現在の推定値で変換行列を作成
        M = create_similarity_transform_matrix(np.rad2deg(theta), scale, 0, 0)
        
        # 入力画像を変換
        I_transformed = apply_transform(I_input, M)
        
        # 微分画像の計算
        Ix, Iy = compute_derivatives(I_transformed)
        
        # 画像差分
        diff = I_transformed - I_output
        
        # 偏微分項の計算
        # ∂T/∂θ = s * [-x*sin(θ) - y*cos(θ), x*cos(θ) - y*sin(θ)]
        # ∂T/∂s = [x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ)]
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # θに関する偏微分
        dT_dtheta_x = -scale * (x_rel * sin_theta + y_rel * cos_theta)
        dT_dtheta_y = scale * (x_rel * cos_theta - y_rel * sin_theta)
        
        # sに関する偏微分  
        dT_ds_x = x_rel * cos_theta - y_rel * sin_theta
        dT_ds_y = x_rel * sin_theta + y_rel * cos_theta
        
        # 勾配項の計算
        grad_theta = Ix * dT_dtheta_x + Iy * dT_dtheta_y
        grad_scale = Ix * dT_ds_x + Iy * dT_ds_y
        
        # マスク内でのみ計算
        valid_mask = mask & (np.abs(Ix) > 1e-6) & (np.abs(Iy) > 1e-6)
        
        if np.sum(valid_mask) < 1000:
            print("有効ピクセルが不足")
            break
            
        # 1階微分
        J_theta = np.sum(diff[valid_mask] * grad_theta[valid_mask])
        J_scale = np.sum(diff[valid_mask] * grad_scale[valid_mask])
        
        # 2階微分（ガウス・ニュートン近似）
        J_theta_theta = np.sum(grad_theta[valid_mask]**2)
        J_scale_scale = np.sum(grad_scale[valid_mask]**2)
        J_theta_scale = np.sum(grad_theta[valid_mask] * grad_scale[valid_mask])
        
        # ヘッセ行列
        H = np.array([
            [J_theta_theta, J_theta_scale],
            [J_theta_scale, J_scale_scale]
        ])
        
        grad_vec = np.array([J_theta, J_scale])
        
        # 条件数チェック
        if np.linalg.cond(H) > 1e10:
            print(f"条件数悪化: {np.linalg.cond(H):.2e}")
            break
            
        # パラメータ更新
        try:
            delta = np.linalg.solve(H, -grad_vec)
            delta_theta, delta_scale = delta
        except np.linalg.LinAlgError:
            print("線形システム解法失敗")
            break
        
        # ステップサイズ制限
        max_theta_step = np.deg2rad(2)  # 2度まで
        max_scale_step = 0.02           # 0.02まで
        
        if abs(delta_theta) > max_theta_step:
            delta_theta = np.sign(delta_theta) * max_theta_step
        if abs(delta_scale) > max_scale_step:
            delta_scale = np.sign(delta_scale) * max_scale_step
        
        # 更新
        theta += delta_theta
        scale += delta_scale
        scale = max(0.1, min(3.0, scale))  # スケール制限
        
        # 目的関数
        objective = 0.5 * np.sum(diff[valid_mask]**2)
        
        # 履歴保存
        theta_history.append(np.rad2deg(theta))
        scale_history.append(scale)
        objective_history.append(objective)
        
        print(f"反復 {iteration+1:2d}: θ = {np.rad2deg(theta):7.3f}°, s = {scale:7.4f}, "
              f"Δθ = {np.rad2deg(delta_theta):7.3f}°, Δs = {delta_scale:7.4f}, "
              f"J = {objective:.2e}")
        
        # 収束判定
        if abs(delta_theta) < threshold and abs(delta_scale) < threshold:
            print(f"収束しました（反復回数: {iteration+1}）")
            break
    
    return {
        'theta': np.rad2deg(theta),
        'scale': scale,
        'theta_history': theta_history,
        'scale_history': scale_history,
        'objective_history': objective_history,
        'iterations': iteration + 1
    }

def visualize_results(results, true_theta=None, true_scale=None):
    """結果の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # θの収束
    axes[0, 0].plot(results['theta_history'], 'b-', linewidth=2, label='Estimated')
    if true_theta is not None:
        axes[0, 0].axhline(y=true_theta, color='r', linestyle='--', linewidth=2, 
                          label=f'True value ({true_theta}°)')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Rotation angle θ (degrees)')
    axes[0, 0].set_title('Convergence of θ')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # sの収束
    axes[0, 1].plot(results['scale_history'], 'g-', linewidth=2, label='Estimated')
    if true_scale is not None:
        axes[0, 1].axhline(y=true_scale, color='r', linestyle='--', linewidth=2,
                          label=f'True value ({true_scale})')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Scale parameter s')
    axes[0, 1].set_title('Convergence of s')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 目的関数
    axes[1, 0].semilogy(results['objective_history'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Objective function J')
    axes[1, 0].set_title('Objective function')
    axes[1, 0].grid(True)
    
    # パラメータ軌跡
    axes[1, 1].plot(results['theta_history'], results['scale_history'], 'o-', 
                   linewidth=2, markersize=3)
    axes[1, 1].plot(results['theta_history'][0], results['scale_history'][0], 
                   'go', markersize=8, label='Initial')
    axes[1, 1].plot(results['theta_history'][-1], results['scale_history'][-1], 
                   'ro', markersize=8, label='Final')
    if true_theta is not None and true_scale is not None:
        axes[1, 1].plot(true_theta, true_scale, 'r*', markersize=12, label='True value')
    axes[1, 1].set_xlabel('Rotation angle θ (degrees)')
    axes[1, 1].set_ylabel('Scale parameter s')
    axes[1, 1].set_title('Parameter trajectory')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('simple_gauss_newton_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("結果をsimple_gauss_newton_results.pngに保存")

def main():
    if len(sys.argv) != 3:
        print("使用方法: python simple_gauss_newton.py input.jpg output.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 画像読み込み
    input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    output_img = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    
    if input_img is None or output_img is None:
        print("画像読み込みエラー")
        sys.exit(1)
    
    # 円形クロップ
    input_cropped = crop_to_circle(input_img)
    output_cropped = crop_to_circle(output_img)
    
    print("=== 簡素化ガウス・ニュートン法による推定 ===")
    
    # 推定実行
    results = gauss_newton_estimation(input_cropped, output_cropped)
    
    print(f"\n=== 最終結果 ===")
    print(f"推定回転角度: {results['theta']:.4f}°")
    print(f"推定スケール: {results['scale']:.4f}")
    print(f"反復回数: {results['iterations']}")
    
    # 可視化（真値を45°, 0.5として表示）
    visualize_results(results, true_theta=45.0, true_scale=0.5)

if __name__ == "__main__":
    main()