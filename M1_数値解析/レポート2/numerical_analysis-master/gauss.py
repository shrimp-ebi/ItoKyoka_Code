"""
【概要】
入力画像と相似変換によって変換した出力画像から回転角度θとスケールパラメータsをガウス・ニュートン法によって推定するプログラム
（表示機能なし版）
"""
import sys
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIバックエンドを無効化
import matplotlib.pyplot as plt
# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
import similarity_transform as st
import os
import pandas as pd
import plot_objective_function as pof



# x方向とy方向に平滑微分フィルタを適用する
def apply_smoothing_differrential_filter(img, kernel_size=3, sigma=1):
    dx_disp = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)  # x方向の微分
    dy_disp = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)  # y方向の微分
    return dx_disp, dy_disp

# ガウスニュートン法によりパラメータを推定する
def estimate_by_gauss_newton_method(img_input, img_output, *, scale_init=1, theta_init=0, threshold=1e-6, max_loop=1000, kernel_size=3, sigma=1):
    # 初期値設定
    theta = np.deg2rad(theta_init)
    scale = scale_init
    I_prime_org = img_input
    I = img_output
    theta_history = []
    scale_history = []
    H, W = I.shape[:2]
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_coords = x_coords - W / 2
    y_coords = y_coords - H / 2
    
    print(f"開始: theta={np.rad2deg(theta):.2f}度, scale={scale:.4f}")
    
    for i in range(max_loop):
        # 推定値を使って画像を相似変換
        M = st.compute_M(scale, theta, 0, 0)
        I_prime = st.apply_similarity_transform_reverse(I_prime_org, M)
        I_prime = st.crop_img_into_circle(I_prime)
        
        I_prime_dx, I_prime_dy = apply_smoothing_differrential_filter(I_prime, kernel_size=kernel_size, sigma=sigma)
        
        # JθとJθθの計算
        dxprime_dtheta = -scale * (x_coords * np.sin(theta) + y_coords * np.cos(theta))
        dyprime_dtheta = scale * (x_coords * np.cos(theta) - y_coords * np.sin(theta))
        J_theta_mat = (I_prime - I) * (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta)
        J_theta = np.sum(J_theta_mat)
        J_theta_theta_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) ** 2 
        J_theta_theta = np.sum(J_theta_theta_mat)
        
        # JSとJSSの計算
        dxprime_dscale = x_coords * np.cos(theta) - y_coords * np.sin(theta)
        dyprime_dscale = x_coords * np.sin(theta) + y_coords * np.cos(theta)
        J_scale_mat = (I_prime - I) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        J_scale = np.sum(J_scale_mat)
        J_scale_scale_mat = (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale) ** 2 
        J_scale_scale = np.sum(J_scale_scale_mat)
        
        # JθSの計算
        J_theta_scale_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        J_theta_scale = np.sum(J_theta_scale_mat)
        objective_func_val = 0.5 * np.sum((I_prime - I) ** 2)

        nabla_u_J = np.array([J_theta, J_scale])
        H_u = np.array([[J_theta_theta, J_theta_scale],
                        [J_theta_scale, J_scale_scale]])
        
        # 行列の条件数をチェック
        if np.linalg.cond(H_u) > 1e12:
            print(f"警告: 反復{i}でヘッセ行列の条件数が悪化しました")
            break
            
        H_u_inv = np.linalg.inv(H_u)
        delta_theta, delta_scale = -H_u_inv @ nabla_u_J
        
        # 収束判定
        if np.abs(delta_theta) < threshold and np.abs(delta_scale) < threshold:
            print(f"収束: 反復{i}, delta_theta:{delta_theta:.2e}, delta_scale:{delta_scale:.2e}")
            break
            
        theta += delta_theta
        scale += delta_scale
        theta_history.append(np.rad2deg(theta))
        scale_history.append(scale)
        
        if i % 10 == 0 or i < 5:  # 進捗表示を減らす
            print(f"反復{i}: theta={np.rad2deg(theta):.4f}度, scale={scale:.6f}, 誤差={objective_func_val:.2e}")
    
    return np.rad2deg(theta), scale, theta_history, scale_history, i

def main():
    # データ準備
    parser = argparse.ArgumentParser(description="ガウス・ニュートン法の実験パラメータ設定")
    parser.add_argument("image_path", type=str, help="入力画像のパス")
    parser.add_argument("scale_true", type=float, help="真値のスケール")
    parser.add_argument("theta_true", type=float, help="真値の角度(deg)")
    parser.add_argument("--scale_init", type=float, default=1, help="初期値のスケール")
    parser.add_argument("--theta_init", type=float, default=0, help="初期値の角度(deg)")
    parser.add_argument("--threshold", type=float, default=1e-6, help="収束判定の閾値")
    parser.add_argument("--max_loop", type=int, default=1000, help="最大反復回数")
    parser.add_argument("--kernel_size", type=int, default=3, help="ガウシアンフィルタのカーネルサイズ")
    parser.add_argument("--sigma", type=float, default=1, help="ガウシアンフィルタのシグマ")
    parser.add_argument("--output_path", type=str, default="output", help="実験結果の出力先のフォルダパス")
    
    args = parser.parse_args()
    img_path = args.image_path
    scale_true = args.scale_true
    theta_true_deg = args.theta_true
    scale_init = args.scale_init
    theta_init_deg = args.theta_init
    threshold = args.threshold
    max_loop = args.max_loop
    kernel_size = args.kernel_size
    sigma = args.sigma
    output_path = args.output_path
    
    print(f"=== 実験設定 ===")
    print(f"画像: {img_path}")
    print(f"真値: スケール={scale_true}, 角度={theta_true_deg}度")
    print(f"初期値: スケール={scale_init}, 角度={theta_init_deg}度")
    print(f"閾値: {threshold}, 最大反復: {max_loop}")
    
    # 画像読み込みと相似変換の適用
    if not os.path.exists(img_path):
        print(f"エラー: 画像ファイル {img_path} が見つかりません")
        return
        
    img_input = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_input is None:
        print(f"エラー: 画像 {img_path} を読み込めません")
        return
        
    img_input_cropped = st.crop_img_into_circle(img_input)
    M = st.compute_M(scale_true, np.deg2rad(theta_true_deg), 0, 0)
    img_output = st.apply_similarity_transform_reverse(img_input, M)
    img_output_cropped = st.crop_img_into_circle(img_output)
    
    # ガウスニュートン法によりパラメータを推定
    theta_est, scale_est, theta_history, scale_history, iteration = estimate_by_gauss_newton_method(
        img_input, img_output_cropped, 
        scale_init=scale_init, 
        theta_init=theta_init_deg, 
        threshold=threshold, 
        max_loop=max_loop, 
        kernel_size=kernel_size, 
        sigma=sigma
    )
    
    print(f"\n=== 推定結果 ===")
    print(f"推定角度: {theta_est:.6f}度 (真値: {theta_true_deg}度, 誤差: {abs(theta_est - theta_true_deg):.6f}度)")
    print(f"推定スケール: {scale_est:.6f} (真値: {scale_true}, 誤差: {abs(scale_est - scale_true):.6f})")
    print(f"反復回数: {iteration}")
    
    # 保存
    img_name = os.path.basename(img_path).split('.')[0]
    output_dir = os.path.join(output_path, f"{img_name}_true_s{scale_true}_t{theta_true_deg}_init_s{scale_init}_t{theta_init_deg}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初期値と推定結果の画像を保存
    M = st.compute_M(scale_init, np.deg2rad(theta_init_deg), 0, 0)
    img_init = st.apply_similarity_transform_reverse(img_input, M)
    img_init_cropped = st.crop_img_into_circle(img_init)
    M = st.compute_M(scale_est, np.deg2rad(theta_est), 0, 0)
    img_est = st.apply_similarity_transform_reverse(img_input, M)
    img_est_cropped = st.crop_img_into_circle(img_est)
    
    # 画像保存
    cv2.imwrite(os.path.join(output_dir, "input.jpg"), img_input_cropped)
    cv2.imwrite(os.path.join(output_dir, "output.jpg"), img_output_cropped)
    cv2.imwrite(os.path.join(output_dir, "init.jpg"), img_init_cropped)
    cv2.imwrite(os.path.join(output_dir, "est.jpg"), img_est_cropped)
    
    # 推定結果の変化をグラフに描画
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(scale_history)
    axs[0].axhline(y=scale_true, color='r', linestyle='--', label=f'True value ({scale_true})')
    axs[0].set_title("Scale History")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Scale")
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot(theta_history)
    axs[1].axhline(y=theta_true_deg, color='r', linestyle='--', label=f'True value ({theta_true_deg}°)')
    axs[1].set_title("Theta History")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Theta (degrees)")
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "scale_theta_history.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 結果をCSV形式で保存
    result_summary = pd.DataFrame([{
        "真値 角度(deg)": theta_true_deg,
        "推定 角度(deg)": theta_est,
        "真値 スケール": scale_true,
        "推定 スケール": scale_est,
        "反復回数": iteration,
        "角度誤差": abs(theta_est - theta_true_deg),
        "スケール誤差": abs(scale_est - scale_true),
        "初期値 角度": theta_init_deg,
        "初期値 スケール": scale_init
    }])
    result_summary.to_csv(os.path.join(output_dir, "result_summary.csv"), index=False, encoding="utf-8-sig")
    
    # 推定結果変化をCSV形式で保存
    if theta_history and scale_history:
        history_length = max(len(theta_history), len(scale_history))
        theta_history_padded = np.pad(theta_history, (0, history_length - len(theta_history)), constant_values=np.nan)
        scale_history_padded = np.pad(scale_history, (0, history_length - len(scale_history)), constant_values=np.nan)
        history_df = pd.DataFrame({
            "iteration": range(history_length),
            "theta_history": theta_history_padded,
            "scale_history": scale_history_padded
        })
        history_df.to_csv(os.path.join(output_dir, "history.csv"), index=False, encoding="utf-8-sig")
    
    print(f"\n結果は {output_dir} に保存されました")

if __name__ == "__main__":
    main()