"""
【概要】
画像を相似変換するプログラム。

【使用方法】
experiment_gauss_newton_method.pyから実行
    # 目的関数を可視化
    # pof.visualize_objective_function(img_input_cropped, img_output_cropped,
    #                                  theta_max=10,
    #                                  theta_min=0,
    #                                  sigma_max=2,
    #                                  simga_min=0.1)
【情報】
作成者：勝田尚樹
作成日：2025/07/23
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import similarity_transform as st
import pandas as pd

# 目的関数を3次元空間にプロットする。極小値確認用。ただし、めっちゃ実行時間かかる
def visualize_objective_function(img_input, img_output, theta_min=0, theta_max=10, theta_step=1, scale_min=0.1, scale_max=2, scale_step=0.1):
    # パラメータ範囲
    I_prime_org = img_input
    I = img_output
    theta_values = np.arange(theta_min, theta_max+theta_step, theta_step)  
    scale_values = np.arange(scale_min, scale_max+scale_step, scale_step)    
    # Jの結果格納用 (scale x theta の2次元配列)
    J_values = np.zeros((len(scale_values), len(theta_values)))
    # I と I_prime は事前に用意されているものとする
    for i, scale in enumerate(scale_values):
        for j, theta in enumerate(theta_values):
            print(f"run theta:{theta}, scale:{scale}")
            # 角度をラジアンに変換
            theta_rad = np.deg2rad(theta)
            # 相似変換を適用
            M = st.compute_M(scale, theta_rad, 0, 0)
            I_prime = st.apply_similarity_transform_reverse(I_prime_org, M)
            I_prime_cropped = st.crop_img_into_circle(I_prime)
            # 目的関数Jを計算
            J = 0.5 * np.sum((I_prime_cropped - I) ** 2)
            J_values[i, j] = J
    # すでに計算済みの J_values, theta_values, scale_values を使用
    Theta, Scale = np.meshgrid(theta_values, scale_values)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # 3Dサーフェスプロット
    surf = ax.plot_surface(Theta, Scale, J_values, cmap='viridis', edgecolor='none')
    # 軸ラベル
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Scale')
    ax.set_zlabel('Objective Function J')
    ax.set_title('3D Plot of J(Theta, Scale)')
    # カラーバー
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='J')
    plt.show()
    # 再描画用にデータを保存
    np.savez(   'output/J_surface_data.npz',
                Theta=Theta,
                Scale=Scale,
                J_values=J_values,
                theta_values=theta_values,
                scale_values=scale_values)

# 目的関数の曲面上での推定値の変化を描画
def visualize_chages_in_est(surface_data_path, history_path):
    # サーフェスデータ（Theta, Scale, J_values）の読み込み
    data = np.load(surface_data_path)
    Theta = data["Theta"]
    Scale = data["Scale"]
    J_values = data["J_values"]
    # 推定履歴の読み込み
    history = pd.read_csv(history_path) 
    theta_hist = history["theta_history"].values
    scale_hist = history["scale_history"].values
    error_hist = history["error_history"].values
    # グラフ準備
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # 曲面描画
    surf = ax.plot_surface(Theta, Scale, J_values, cmap='viridis', edgecolor='none', alpha=0.5)
    # ax.plot_wireframe(Theta, Scale, J_values, color='gray', linewidth=0.5, alpha=1)
    # 推定値描画
    ax.plot(theta_hist, scale_hist, error_hist,
            color='red', marker='o', label='History (Error)')
    # ラベル設定
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Scale')
    ax.set_zlabel('Objective Function J')
    ax.set_title('3D Plot of J(Theta, Scale) with History')

    ax.legend()
    plt.show()

def main():
    # scale_true = 1.2
    # theta_true_deg = 45 
    # img_path = "input/color/Lenna.bmp"
    # img_input = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img_input_cropped = st.crop_img_into_circle(img_input)
    # M = st.compute_M(scale_true, np.deg2rad(theta_true_deg), 0, 0)
    # img_output = st.apply_similarity_transform_reverse(img_input, M)
    # img_output_cropped = st.crop_img_into_circle(img_output)
    # # 目的関数を可視化
    # visualize_objective_function(img_input_cropped, img_output_cropped,
    #                                  theta_min=35,
    #                                  theta_max=55,
    #                                  theta_step=1,
    #                                  scale_min=1,
    #                                  scale_max=1.4,
    #                                  scale_step=0.05)
    
    surface_data_path = "output/J_surface_data.npz"
    history_path = "output\Lenna.bmp_true_s1.2_t45.0_init_s1.1_t45.0\history.csv"
    visualize_chages_in_est(surface_data_path, history_path)
                                     

if __name__ == "__main__":
    main()