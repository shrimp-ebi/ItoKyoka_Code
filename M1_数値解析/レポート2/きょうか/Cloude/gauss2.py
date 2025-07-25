"""
ガウス・ニュートン法による相似変換パラメータ推定プログラム（改良版）
Yu-gauss.pyにより忠実な実装

使用方法:
python gauss2.py --input Fuji.jpg --output out-Fuji-gray.png
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# 試行回数
loop = 100
# 閾値
threshold = 1e-5

def create_gaussian_filters(sigma=1):
    """ガウシアンフィルタの作成"""
    coef = 1 / (2 * np.pi * (sigma**2))
    size = int(1 + 4 * sigma)
    gaussian_x = np.zeros((size, size))
    gaussian_y = np.zeros((size, size))
    
    for y in range(size):
        offset_y = round(size / 2) - y
        for x in range(size):
            offset_x = x - round(size / 2)
            exp_val = np.exp(-1 * (offset_x**2 + offset_y**2) / (2 * sigma**2))
            gaussian_x[y, x] = -coef * (-offset_x / sigma**2) * exp_val
            gaussian_y[y, x] = -coef * (-offset_y / sigma**2) * exp_val
    
    return gaussian_x, gaussian_y

def main():
    parser = argparse.ArgumentParser(description='ガウス・ニュートン法による相似変換パラメータ推定')
    parser.add_argument('--input', '-i', type=str, default='Fuji.jpg',
                        help='入力画像ファイル名')
    parser.add_argument('--output', '-o', type=str, default='out-Fuji-gray.png',
                        help='出力画像ファイル名')
    parser.add_argument('--theta_init', type=float, default=0,
                        help='回転角度の初期値（度数法）')
    parser.add_argument('--scale_init', type=float, default=1,
                        help='スケールの初期値')
    
    args = parser.parse_args()
    
    # 初期パラメータ
    theta = np.deg2rad(args.theta_init)
    s = args.scale_init
    
    # 履歴保存用
    all_s = []
    all_theta = []
    all_error = []
    all_J_theta = []
    all_J_s = []
    
    # ガウシアンフィルタの作成
    gaussian_x, gaussian_y = create_gaussian_filters(sigma=1)
    
    # 画像の読み込み
    img_in = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img_out = cv2.imread(args.output, cv2.IMREAD_GRAYSCALE)
    
    if img_in is None or img_out is None:
        print("エラー: 画像を読み込めません")
        return
    
    print(f"入力画像: {args.input} (サイズ: {img_in.shape})")
    print(f"出力画像: {args.output} (サイズ: {img_out.shape})")
    print(f"初期値: θ={np.rad2deg(theta):.2f}°, s={s:.4f}")
    
    # 画像の大きさ
    pixel = img_in.shape[0]
    
    # 画像中心
    cx_in = img_in.shape[0] / 2
    cy_in = cx_in
    
    # 有効ピクセルの座標を事前計算
    X = []
    Y = []
    radius = img_in.shape[0] / 2
    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            length = np.sqrt((i - cx_in) ** 2 + (j - cy_in) ** 2)
            if length <= radius:
                X.append(i)
                Y.append(j)
    
    # 作業用配列
    im = np.zeros(img_in.shape, dtype=float)
    Ix = np.zeros(img_in.shape, dtype=float)
    Iy = np.zeros(img_in.shape, dtype=float)
    
    # ガウス・ニュートン法のメインループ
    for i in range(loop):
        J_theta = 0
        J_2theta = 0
        J_s = 0
        J_2s = 0
        J_theta_s = 0
        error = 0
        
        # 現パラメータのθ分だけ回転させる（逆変換）
        for j in range(len(X)):
            x = X[j]
            y = Y[j]
            x_rot = round(
                ((x - cx_in) * np.cos(-theta) - (y - cy_in) * np.sin(-theta)) + cx_in
            )
            y_rot = round(
                ((x - cx_in) * np.sin(-theta) + (y - cy_in) * np.cos(-theta)) + cy_in
            )
            if x_rot >= pixel or y_rot >= pixel or x_rot < 0 or y_rot < 0:
                im[y, x] = 0
            else:
                im[y, x] = img_out[y_rot, x_rot]
        
        # 平滑微分計算
        im = im.astype(np.float64)
        Ixt = cv2.filter2D(im, -1, gaussian_x)
        Iyt = cv2.filter2D(im, -1, gaussian_y)
        
        # 画像を元の座標系に戻す
        for j in range(len(X)):
            x = X[j]
            y = Y[j]
            x_rot = round(
                ((x - cx_in) * np.cos(theta) - (y - cy_in) * np.sin(theta)) + cx_in
            )
            y_rot = round(
                ((x - cx_in) * np.sin(theta) + (y - cy_in) * np.cos(theta)) + cy_in
            )
            if x_rot >= pixel or y_rot >= pixel or x_rot < 0 or y_rot < 0:
                Ix[y, x] = 0
                Iy[y, x] = 0
            else:
                Ix[y, x] = Ixt[y_rot, x_rot]
                Iy[y, x] = Iyt[y_rot, x_rot]
        
        # 勾配計算
        for j in range(len(X)):
            x = X[j]
            y = Y[j]
            
            # 変換後の座標（Yu-gauss.pyと同じ）
            dif_x = round(
                s * ((x - cx_in) * np.cos(theta) - (cy_in - y) * np.sin(theta)) + cx_in
            )
            dif_y = round(
                -s * ((x - cx_in) * np.sin(theta) + (cy_in - y) * np.cos(theta)) + cy_in
            )
            
            if dif_x >= pixel or dif_y >= pixel or dif_x < 0 or dif_y < 0:
                dif_I = 0
                im[y, x] = 0
                dif_Ix = 0
                dif_Iy = 0
            else:
                dif_I = img_out[dif_y, dif_x]
                im[y, x] = img_out[dif_y, dif_x]
                dif_Ix = Ix[dif_y, dif_x]
                dif_Iy = Iy[dif_y, dif_x]
                # オーバーフロー防止
                dif_I = dif_I.astype("int64")
            
            I = img_in[y, x]
            I = I.astype("int64")
            
            # x,yをthetaで微分
            dx_theta = s * (-(x - cx_in) * np.sin(theta) - (cy_in - y) * np.cos(theta))
            dy_theta = s * ((x - cx_in) * np.cos(theta) - (cy_in - y) * np.sin(theta))
            
            # x,yをsで微分
            dx_s = (x - cx_in) * np.cos(theta) - (cy_in - y) * np.sin(theta)
            dy_s = (x - cx_in) * np.sin(theta) + (cy_in - y) * np.cos(theta)
            
            # thetaでの微分式
            J_theta = J_theta + ((dif_I - I) * (dif_Ix * dx_theta + dif_Iy * dy_theta))
            J_2theta = J_2theta + (dif_Ix * dx_theta + dif_Iy * dy_theta) ** 2
            
            # sでの微分式
            J_s = J_s + ((dif_I - I) * (dif_Ix * dx_s + dif_Iy * dy_s))
            J_2s = J_2s + (dif_Ix * dx_s + dif_Iy * dy_s) ** 2
            
            # thetaとsで微分
            J_theta_s = (
                J_theta_s
                + dif_Ix**2 * dx_theta * dx_s
                + dif_Ix * dif_Iy * (dx_theta * dy_s + dx_s * dy_theta)
                + dif_Iy**2 * dy_theta * dy_s
            )
            
            error = error + (1 / 2) * (dif_I - I) ** 2
        
        # 履歴の保存
        all_theta.append(np.rad2deg(theta))
        all_s.append(s)
        all_error.append(error)
        all_J_theta.append(J_theta)
        all_J_s.append(J_s)
        
        # ヘッセ行列と勾配ベクトル
        J = np.array([[J_2theta, J_theta_s], [J_theta_s, J_2s]])
        J_vec = np.array([[J_theta], [J_s]])
        
        # パラメータ更新量の計算
        try:
            [dtheta, ds] = np.matmul(np.linalg.inv(J), J_vec)
        except np.linalg.LinAlgError:
            print("ヘッセ行列が特異です")
            break
        
        # 収束判定
        if np.abs(dtheta[0]) < threshold and np.abs(ds[0]) < threshold:
            print(f"収束しました（反復回数: {i+1}）")
            break
        
        # パラメータ更新
        theta = theta - dtheta[0]
        s = s - ds[0]
        
        print(f"反復 {i+1}: θ={np.rad2deg(theta):.4f}°, s={s:.4f}, "
              f"誤差={error:.2f}, |Δθ|={abs(dtheta[0]):.2e}, |Δs|={abs(ds[0]):.2e}")
    
    print(f"\n推定結果: θ={np.rad2deg(theta):.4f}°, s={s:.4f}")
    
    # 収束グラフの作成
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # θの推定過程
    ax = axes[0, 0]
    ax.plot(all_theta, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('θ [degrees]')
    ax.set_title('Rotation Angle Estimation')
    ax.grid(True)
    
    # sの推定過程
    ax = axes[0, 1]
    ax.plot(all_s, 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Scale')
    ax.set_title('Scale Parameter Estimation')
    ax.grid(True)
    
    # 誤差の推移
    ax = axes[0, 2]
    ax.semilogy(all_error, 'g-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    ax.set_title('Total Error')
    ax.grid(True)
    
    # θの1階微分
    ax = axes[1, 0]
    ax.plot(all_J_theta, 'b--', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('∂J/∂θ')
    ax.set_title('Gradient w.r.t. θ')
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # sの1階微分
    ax = axes[1, 1]
    ax.plot(all_J_s, 'r--', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('∂J/∂s')
    ax.set_title('Gradient w.r.t. s')
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # パラメータの変化（Δθ, Δs）
    ax = axes[1, 2]
    if len(all_theta) > 1:
        delta_theta = np.abs(np.diff(all_theta))
        delta_s = np.abs(np.diff(all_s))
        ax.semilogy(delta_theta, 'b-', label='|Δθ|', linewidth=2)
        ax.semilogy(delta_s, 'r-', label='|Δs|', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Change')
        ax.set_title('Convergence of Parameters')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_plots2.png', dpi=150)
    plt.close()
    print("\n収束グラフを 'convergence_plots2.png' に保存しました")
    
    # CSVに保存
    import csv
    with open('estimation_history2.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'theta', 'scale', 'error', 'J_theta', 'J_s'])
        for i in range(len(all_theta)):
            writer.writerow([i, all_theta[i], all_s[i], all_error[i], 
                           all_J_theta[i] if i < len(all_J_theta) else 0,
                           all_J_s[i] if i < len(all_J_s) else 0])
    print("推定履歴を 'estimation_history2.csv' に保存しました")

if __name__ == '__main__':
    main()