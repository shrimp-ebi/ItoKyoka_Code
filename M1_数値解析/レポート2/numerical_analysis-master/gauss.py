# -*- coding: utf-8 -*-
"""
【概要】
入力画像と相似変換によって変換した出力画像から
回転角度θとスケールsをガウス・ニュートン法で推定する（表示機能なし版）
"""
import sys
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
import similarity_transform as st
import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # GUI非表示
import matplotlib.pyplot as plt

# フォント設定（Windows用）
plt.rcParams['font.family'] = 'MS Gothic'  # または 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False


def apply_smoothing_differrential_filter(img, kernel_size=3, sigma=1):
    dx_disp = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    dy_disp = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    return dx_disp, dy_disp

def estimate_by_gauss_newton_method(img_input, img_output, *, scale_init=1, theta_init=0, threshold=1e-6, max_loop=1000, kernel_size=3, sigma=1):
    theta = np.deg2rad(theta_init)
    scale = scale_init
    I_prime_org = img_input
    I = img_output

    # 初期化
    theta_history = []
    scale_history = []
    J_theta_list = []
    J_scale_list = []
    J_value_list = []

    H, W = I.shape[:2]
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_coords = x_coords - W / 2
    y_coords = y_coords - H / 2

    print(f"開始: theta={np.rad2deg(theta):.2f}度, scale={scale:.4f}")

    for i in range(max_loop):
        M = st.compute_M(scale, theta, 0, 0)
        I_prime = st.apply_similarity_transform_reverse(I_prime_org, M)
        I_prime = st.crop_img_into_circle(I_prime)

        I_prime_dx, I_prime_dy = apply_smoothing_differrential_filter(I_prime, kernel_size=kernel_size, sigma=sigma)

        dxprime_dtheta = -scale * (x_coords * np.sin(theta) + y_coords * np.cos(theta))
        dyprime_dtheta = scale * (x_coords * np.cos(theta) - y_coords * np.sin(theta))
        J_theta_mat = (I_prime - I) * (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta)
        J_theta = np.sum(J_theta_mat)
        J_theta_theta_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) ** 2
        J_theta_theta = np.sum(J_theta_theta_mat)

        dxprime_dscale = x_coords * np.cos(theta) - y_coords * np.sin(theta)
        dyprime_dscale = x_coords * np.sin(theta) + y_coords * np.cos(theta)
        J_scale_mat = (I_prime - I) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        J_scale = np.sum(J_scale_mat)
        J_scale_scale_mat = (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale) ** 2
        J_scale_scale = np.sum(J_scale_scale_mat)

        J_theta_scale_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        J_theta_scale = np.sum(J_theta_scale_mat)
        objective_func_val = 0.5 * np.sum((I_prime - I) ** 2)

        nabla_u_J = np.array([J_theta, J_scale])
        H_u = np.array([[J_theta_theta, J_theta_scale],
                        [J_theta_scale, J_scale_scale]])

        if np.linalg.cond(H_u) > 1e12:
            print(f"警告: 反復{i}でヘッセ行列の条件数が悪化しました")
            break

        H_u_inv = np.linalg.inv(H_u)
        delta_theta, delta_scale = -H_u_inv @ nabla_u_J

        if np.abs(delta_theta) < threshold and np.abs(delta_scale) < threshold:
            print(f"収束: 反復{i}, delta_theta:{delta_theta:.2e}, delta_scale:{delta_scale:.2e}")
            break

        theta += delta_theta
        scale += delta_scale
        theta_history.append(np.rad2deg(theta))
        scale_history.append(scale)
        J_theta_list.append(J_theta)
        J_scale_list.append(J_scale)
        J_value_list.append(objective_func_val)


        if i % 10 == 0 or i < 5:
            print(f"反復{i}: theta={np.rad2deg(theta):.4f}度, scale={scale:.6f}, 誤差={objective_func_val:.2e}")

    return np.rad2deg(theta), scale, theta_history, scale_history, i, J_theta_list, J_scale_list, J_value_list

def main():
    parser = argparse.ArgumentParser(description="ガウス・ニュートン法による回転角度とスケール推定")
    parser.add_argument("image_path", type=str, help="入力画像のパス")
    parser.add_argument("scale_true", type=float, help="真のスケール値")
    parser.add_argument("theta_true", type=float, help="真の回転角度（度）")
    parser.add_argument("--scale_init", type=float, default=1, help="初期スケール値")
    parser.add_argument("--theta_init", type=float, default=0, help="初期角度（度）")
    parser.add_argument("--threshold", type=float, default=1e-6, help="収束判定の閾値")
    parser.add_argument("--max_loop", type=int, default=1000, help="最大反復回数")
    parser.add_argument("--kernel_size", type=int, default=3, help="ガウシアンフィルタのカーネルサイズ")
    parser.add_argument("--sigma", type=float, default=1, help="ガウシアンフィルタのシグマ")
    parser.add_argument("--output_path", type=str, default="output", help="結果保存ディレクトリ")
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

    theta_est, scale_est, theta_history, scale_history, iteration, J_theta_list, J_scale_list, J_value_list = estimate_by_gauss_newton_method(
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

    img_name = os.path.basename(img_path).split('.')[0]
    output_dir = os.path.join(output_path, f"{img_name}_true_s{scale_true}_t{theta_true_deg}_init_s{scale_init}_t{theta_init_deg}")
    os.makedirs(output_dir, exist_ok=True)

    M = st.compute_M(scale_init, np.deg2rad(theta_init_deg), 0, 0)
    img_init = st.apply_similarity_transform_reverse(img_input, M)
    img_init_cropped = st.crop_img_into_circle(img_init)
    M = st.compute_M(scale_est, np.deg2rad(theta_est), 0, 0)
    img_est = st.apply_similarity_transform_reverse(img_input, M)
    img_est_cropped = st.crop_img_into_circle(img_est)

    cv2.imwrite(os.path.join(output_dir, "input.jpg"), img_input_cropped)
    cv2.imwrite(os.path.join(output_dir, "output.jpg"), img_output_cropped)
    cv2.imwrite(os.path.join(output_dir, "init.jpg"), img_init_cropped)
    cv2.imwrite(os.path.join(output_dir, "est.jpg"), img_est_cropped)

    # θ履歴（回転角度）グラフ（日本語）
    plt.figure(figsize=(6, 4))
    plt.plot(theta_history)
    plt.axhline(y=theta_true_deg, color='r', linestyle='--', label=f'真値 {theta_true_deg}°')
    plt.title("回転角度θの推定履歴")
    plt.xlabel("反復回数")
    plt.ylabel("角度θ（度）")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "theta_history.png"), dpi=300)
    plt.close()

    # スケール履歴グラフ（日本語）
    plt.figure(figsize=(6, 4))
    plt.plot(scale_history)
    plt.axhline(y=scale_true, color='r', linestyle='--', label=f'真値 {scale_true}')
    plt.title("スケールsの推定履歴")
    plt.xlabel("反復回数")
    plt.ylabel("スケールs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scale_history.png"), dpi=300)
    plt.close()

    # Jの勾配（∂J/∂θ, ∂J/∂s）グラフ
    plt.figure(figsize=(6, 4))
    plt.plot(J_theta_list, label='∂J/∂θ')
    plt.plot(J_scale_list, label='∂J/∂s')
    plt.title("目的関数Jの勾配推移")
    plt.xlabel("反復回数")
    plt.ylabel("勾配値")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_history.png"), dpi=300)
    plt.close()

    # 目的関数Jの推移
    plt.figure(figsize=(6, 4))
    plt.plot(J_value_list, label='目的関数J')
    plt.title("目的関数Jの推移")
    plt.xlabel("反復回数")
    plt.ylabel("J")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "objective_function_history.png"), dpi=300)
    plt.close()

    # 新規追加: 回転角度に対する目的関数の変化（真値中心）
    theta_range = np.linspace(theta_true_deg - 10, theta_true_deg + 10, 100)
    J_theta_range = []
    
    for theta_test in theta_range:
        M_test = st.compute_M(scale_true, np.deg2rad(theta_test), 0, 0)
        img_test = st.apply_similarity_transform_reverse(img_input, M_test)
        img_test_cropped = st.crop_img_into_circle(img_test)
        J_test = 0.5 * np.sum((img_test_cropped - img_output_cropped) ** 2)
        J_theta_range.append(J_test)
    
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range, J_theta_range, 'b-', linewidth=2, label='目的関数J')
    plt.axvline(x=theta_true_deg, color='r', linestyle='--', linewidth=2, label=f'真値 {theta_true_deg}°')
    plt.axvline(x=theta_est, color='g', linestyle=':', linewidth=2, label=f'推定値 {theta_est:.2f}°')
    plt.title("回転角度θに対する目的関数Jの変化")
    plt.xlabel("回転角度θ（度）")
    plt.ylabel("目的関数J")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "objective_vs_theta.png"), dpi=300)
    plt.close()

    # 新規追加: 回転角度θに対する目的関数の一階微分 ∂J/∂θ
    theta_range_grad = np.linspace(theta_true_deg - 10, theta_true_deg + 10, 100)
    dJ_dtheta_range = []
    
    for theta_test in theta_range_grad:
        M_test = st.compute_M(scale_true, np.deg2rad(theta_test), 0, 0)
        I_prime_test = st.apply_similarity_transform_reverse(img_input, M_test)
        I_prime_test = st.crop_img_into_circle(I_prime_test)
        
        I_prime_dx, I_prime_dy = apply_smoothing_differrential_filter(I_prime_test, kernel_size=kernel_size, sigma=sigma)
        
        H, W = I_prime_test.shape[:2]
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_coords = x_coords - W / 2
        y_coords = y_coords - H / 2
        
        dxprime_dtheta = -scale_true * (x_coords * np.sin(np.deg2rad(theta_test)) + y_coords * np.cos(np.deg2rad(theta_test)))
        dyprime_dtheta = scale_true * (x_coords * np.cos(np.deg2rad(theta_test)) - y_coords * np.sin(np.deg2rad(theta_test)))
        
        J_theta_grad = np.sum((I_prime_test - img_output_cropped) * (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta))
        dJ_dtheta_range.append(J_theta_grad)
    
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range_grad, dJ_dtheta_range, 'purple', linewidth=2, label='∂J/∂θ')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=theta_true_deg, color='r', linestyle='--', linewidth=2, label=f'真値 {theta_true_deg}°')
    plt.axvline(x=theta_est, color='g', linestyle=':', linewidth=2, label=f'推定値 {theta_est:.2f}°')
    plt.title("回転角度θに対する目的関数の一階微分 ∂J/∂θ")
    plt.xlabel("回転角度θ（度）")
    plt.ylabel("∂J/∂θ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_vs_theta.png"), dpi=300)
    plt.close()

    # 新規追加: スケールsに対する目的関数の一階微分 ∂J/∂s
    scale_range_grad = np.linspace(max(0.1, scale_true - 0.3), scale_true + 0.3, 100)
    dJ_dscale_range = []
    
    for scale_test in scale_range_grad:
        M_test = st.compute_M(scale_test, np.deg2rad(theta_true_deg), 0, 0)
        I_prime_test = st.apply_similarity_transform_reverse(img_input, M_test)
        I_prime_test = st.crop_img_into_circle(I_prime_test)
        
        I_prime_dx, I_prime_dy = apply_smoothing_differrential_filter(I_prime_test, kernel_size=kernel_size, sigma=sigma)
        
        H, W = I_prime_test.shape[:2]
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_coords = x_coords - W / 2
        y_coords = y_coords - H / 2
        
        dxprime_dscale = x_coords * np.cos(np.deg2rad(theta_true_deg)) - y_coords * np.sin(np.deg2rad(theta_true_deg))
        dyprime_dscale = x_coords * np.sin(np.deg2rad(theta_true_deg)) + y_coords * np.cos(np.deg2rad(theta_true_deg))
        
        J_scale_grad = np.sum((I_prime_test - img_output_cropped) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale))
        dJ_dscale_range.append(J_scale_grad)
    
    plt.figure(figsize=(8, 5))
    plt.plot(scale_range_grad, dJ_dscale_range, 'orange', linewidth=2, label='∂J/∂s')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(x=scale_true, color='r', linestyle='--', linewidth=2, label=f'真値 {scale_true}')
    plt.axvline(x=scale_est, color='g', linestyle=':', linewidth=2, label=f'推定値 {scale_est:.3f}')
    plt.title("スケールsに対する目的関数の一階微分 ∂J/∂s")
    plt.xlabel("スケールs")
    plt.ylabel("∂J/∂s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_vs_scale.png"), dpi=300)
    plt.close()

    # 新規追加: スケールに対する目的関数の変化（真値中心）
    scale_range = np.linspace(max(0.1, scale_true - 0.3), scale_true + 0.3, 100)
    J_scale_range = []
    
    for scale_test in scale_range:
        M_test = st.compute_M(scale_test, np.deg2rad(theta_true_deg), 0, 0)
        img_test = st.apply_similarity_transform_reverse(img_input, M_test)
        img_test_cropped = st.crop_img_into_circle(img_test)
        J_test = 0.5 * np.sum((img_test_cropped - img_output_cropped) ** 2)
        J_scale_range.append(J_test)
    
    plt.figure(figsize=(8, 5))
    plt.plot(scale_range, J_scale_range, 'b-', linewidth=2, label='目的関数J')
    plt.axvline(x=scale_true, color='r', linestyle='--', linewidth=2, label=f'真値 {scale_true}')
    plt.axvline(x=scale_est, color='g', linestyle=':', linewidth=2, label=f'推定値 {scale_est:.3f}')
    plt.title("スケールsに対する目的関数Jの変化")
    plt.xlabel("スケールs")
    plt.ylabel("目的関数J")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "objective_vs_scale.png"), dpi=300)
    plt.close()

    # 新規追加: 真値からの誤差の推移グラフ
    theta_error_history = [abs(theta - theta_true_deg) for theta in theta_history]
    scale_error_history = [abs(scale - scale_true) for scale in scale_history]
    
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.plot(theta_error_history, 'b-', linewidth=2)
    plt.title("回転角度の誤差推移")
    plt.xlabel("反復回数")
    plt.ylabel("角度誤差（度）")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(scale_error_history, 'r-', linewidth=2)
    plt.title("スケールの誤差推移")
    plt.xlabel("反復回数")
    plt.ylabel("スケール誤差")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_convergence.png"), dpi=300)
    plt.close()

   

    # 3. 目的関数の一階微分（勾配ノルム）の変化
    if J_theta_list and J_scale_list:
        gradient_norms = [np.sqrt(jt**2 + js**2) for jt, js in zip(J_theta_list, J_scale_list)]
        gradient_iterations = list(range(len(gradient_norms)))
        
        plt.figure(figsize=(8, 6))
        plt.plot(gradient_norms, gradient_iterations, 'purple', marker='o', markersize=3, linewidth=2, label='勾配ノルム')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='収束目標（勾配=0）')
        plt.title("目的関数の一階微分（勾配ノルム）の変化")
        plt.xlabel("勾配ノルム ||∇J||")
        plt.ylabel("反復回数")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.xscale('log')  # 対数スケール
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gradient_norm_path.png"), dpi=300)
        plt.close()

    


    # CSV保存
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

    # 勾配と目的関数の履歴をCSVで保存
    if J_theta_list and J_scale_list and J_value_list:
        grad_df = pd.DataFrame({
            "反復回数": list(range(len(J_theta_list))),
            "∂J/∂θ": J_theta_list,
            "∂J/∂s": J_scale_list,
            "目的関数J": J_value_list
        })
        grad_df.to_csv(os.path.join(output_dir, "gradient_and_objective_history.csv"), index=False, encoding="utf-8-sig")


    if theta_history and scale_history:
        history_df = pd.DataFrame({
            "反復回数": list(range(len(theta_history))),
            "回転角度θ（deg）": theta_history,
            "スケールs": scale_history
        })
        history_df.to_csv(os.path.join(output_dir, "history.csv"), index=False, encoding="utf-8-sig")
        
        # 新規追加: 解析用データのCSV保存
        analysis_df = pd.DataFrame({
            "回転角度θ（度）": theta_range,
            "目的関数J（θ変化）": J_theta_range
        })
        analysis_df.to_csv(os.path.join(output_dir, "theta_analysis.csv"), index=False, encoding="utf-8-sig")
        
        scale_analysis_df = pd.DataFrame({
            "スケールs": scale_range,
            "目的関数J（s変化）": J_scale_range
        })
        scale_analysis_df.to_csv(os.path.join(output_dir, "scale_analysis.csv"), index=False, encoding="utf-8-sig")
        
        # 勾配解析データの保存
        theta_grad_df = pd.DataFrame({
            "回転角度θ（度）": theta_range_grad,
            "∂J/∂θ": dJ_dtheta_range
        })
        theta_grad_df.to_csv(os.path.join(output_dir, "theta_gradient_analysis.csv"), index=False, encoding="utf-8-sig")
        
        scale_grad_df = pd.DataFrame({
            "スケールs": scale_range_grad,
            "∂J/∂s": dJ_dscale_range
        })
        scale_grad_df.to_csv(os.path.join(output_dir, "scale_gradient_analysis.csv"), index=False, encoding="utf-8-sig")
        
        # 新規追加: パラメータ軌跡データの保存
        if theta_history and scale_history and J_value_list and J_theta_list and J_scale_list:
            gradient_norms = [np.sqrt(jt**2 + js**2) for jt, js in zip(J_theta_list, J_scale_list)]
            path_df = pd.DataFrame({
                "反復回数": list(range(len(theta_history))),
                "回転角度θ（度）": theta_history,
                "スケールs": scale_history,
                "目的関数値J": J_value_list,
                "勾配ノルム": gradient_norms,
                "∂J/∂θ": J_theta_list,
                "∂J/∂s": J_scale_list
            })
            path_df.to_csv(os.path.join(output_dir, "solution_path.csv"), index=False, encoding="utf-8-sig")

    print(f"\n結果は {output_dir} に保存されました")

if __name__ == "__main__":
    main()
