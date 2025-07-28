"""
ガウス・ニュートン法による相似変換パラメータ推定プログラム
入力画像と出力画像から回転角度θとスケールsを推定する

使用方法:
python gauss.py -i input.jpg -o output.png --init_theta 0 --init_scale 1
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # バックエンドを非表示モードに設定
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
import os

class GaussNewtonEstimator:
    def __init__(self, img_in, img_out, sigma=1.0):
        """
        Parameters:
        img_in: 入力画像（グレースケール）
        img_out: 出力画像（グレースケール）
        sigma: ガウシアンフィルタのσ値
        """
        self.img_in = img_in.astype(np.float64)
        self.img_out = img_out.astype(np.float64)
        self.height, self.width = img_in.shape
        
        # 画像中心
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        
        # 使用する領域（円内）のピクセル座標を事前計算
        self.radius = min(self.width, self.height) / 2.0
        self.X, self.Y = self._get_valid_pixels()
        
        # ガウシアン微分フィルタの作成
        self.gaussian_x, self.gaussian_y = self._create_gaussian_derivative_filters(sigma)
        
        # 記録用リスト
        self.theta_history = []
        self.scale_history = []
        self.error_history = []
        self.J_theta_history = []
        self.J_s_history = []
        self.delta_theta_history = []
        self.delta_s_history = []
        
    def _get_valid_pixels(self):
        """円内の有効なピクセル座標を取得"""
        X = []
        Y = []
        for y in range(self.height):
            for x in range(self.width):
                distance = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
                if distance <= self.radius:
                    X.append(x)
                    Y.append(y)
        return np.array(X), np.array(Y)
    
    def _create_gaussian_derivative_filters(self, sigma):
        """ガウシアン微分フィルタを作成"""
        kernel_size = int(1 + 4 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        gaussian_x = np.zeros((kernel_size, kernel_size))
        gaussian_y = np.zeros((kernel_size, kernel_size))
        
        center = kernel_size // 2
        coef = 1 / (2 * np.pi * sigma**2)
        
        for y in range(kernel_size):
            for x in range(kernel_size):
                offset_x = x - center
                offset_y = y - center
                exp_term = np.exp(-(offset_x**2 + offset_y**2) / (2 * sigma**2))
                
                gaussian_x[y, x] = -coef * (-offset_x / sigma**2) * exp_term
                gaussian_y[y, x] = -coef * (-offset_y / sigma**2) * exp_term
                
        return gaussian_x, gaussian_y
    
    def _compute_transformed_coords(self, x, y, theta, s):
        """座標(x,y)を相似変換"""
        x_centered = x - self.cx
        y_centered = y - self.cy
        
        x_transformed = s * (x_centered * np.cos(theta) - y_centered * np.sin(theta)) + self.cx
        y_transformed = s * (x_centered * np.sin(theta) + y_centered * np.cos(theta)) + self.cy
        
        return x_transformed, y_transformed
    
    def _interpolate_image(self, img, x, y):
        """バイリニア補間"""
        h, w = img.shape
        
        # 境界チェック
        if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
            return 0
        
        # 整数部分と小数部分
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0, y - y0
        
        # バイリニア補間
        value = (1 - dx) * (1 - dy) * img[y0, x0] + \
                dx * (1 - dy) * img[y0, x1] + \
                (1 - dx) * dy * img[y1, x0] + \
                dx * dy * img[y1, x1]
                
        return value
    
    def estimate(self, init_theta=0, init_s=1.0, max_iter=100, threshold=1e-5, 
                 threshold_theta=None, threshold_scale=None):
        """
        ガウス・ニュートン法でパラメータを推定
        
        Parameters:
        init_theta: 初期回転角度（ラジアン）
        init_s: 初期スケール
        max_iter: 最大反復回数
        threshold: 収束判定閾値（デフォルト）
        threshold_theta: 回転角度の収束判定閾値（度）
        threshold_scale: スケールの収束判定閾値
        """
        theta = init_theta
        s = init_s
        
        # 個別の閾値が指定されていない場合はデフォルトを使用
        if threshold_theta is None:
            threshold_theta = threshold
        else:
            threshold_theta = np.deg2rad(threshold_theta)  # 度からラジアンに変換
        
        if threshold_scale is None:
            threshold_scale = threshold
        
        print(f"初期値: θ={np.rad2deg(theta):.2f}度, s={s:.4f}")
        
        for iteration in range(max_iter):
            # 1階微分と2階微分の初期化
            J_theta = 0
            J_theta_theta = 0
            J_s = 0
            J_s_s = 0
            J_theta_s = 0
            error = 0
            
            # 出力画像の平滑微分画像を作成
            Ix = cv2.filter2D(self.img_out, -1, self.gaussian_x)
            Iy = cv2.filter2D(self.img_out, -1, self.gaussian_y)
            
            # 各ピクセルでの計算
            for i in range(len(self.X)):
                x, y = self.X[i], self.Y[i]
                
                # 変換後の座標
                x_transformed, y_transformed = self._compute_transformed_coords(x, y, theta, s)
                
                # 出力画像の値（補間）
                I_out = self._interpolate_image(self.img_out, x_transformed, y_transformed)
                I_in = self.img_in[int(y), int(x)]
                
                # 画像勾配（補間）
                Ix_val = self._interpolate_image(Ix, x_transformed, y_transformed)
                Iy_val = self._interpolate_image(Iy, x_transformed, y_transformed)
                
                # 座標の微分
                x_c = x - self.cx
                y_c = y - self.cy
                
                # θに関する微分
                dx_dtheta = -s * (x_c * np.sin(theta) + y_c * np.cos(theta))
                dy_dtheta = s * (x_c * np.cos(theta) - y_c * np.sin(theta))
                
                # sに関する微分
                dx_ds = x_c * np.cos(theta) - y_c * np.sin(theta)
                dy_ds = x_c * np.sin(theta) + y_c * np.cos(theta)
                
                # 誤差
                diff = I_out - I_in
                error += 0.5 * diff**2
                
                # 1階微分
                grad_theta = Ix_val * dx_dtheta + Iy_val * dy_dtheta
                grad_s = Ix_val * dx_ds + Iy_val * dy_ds
                
                J_theta += diff * grad_theta
                J_s += diff * grad_s
                
                # 2階微分（ガウス・ニュートン近似）
                J_theta_theta += grad_theta**2
                J_s_s += grad_s**2
                J_theta_s += grad_theta * grad_s
            
            # ヘッセ行列と勾配ベクトル
            H = np.array([[J_theta_theta, J_theta_s],
                         [J_theta_s, J_s_s]])
            g = np.array([J_theta, J_s])
            
            # パラメータ更新量
            try:
                delta = np.linalg.solve(H, g)
                delta_theta, delta_s = delta[0], delta[1]
            except np.linalg.LinAlgError:
                print("警告: ヘッセ行列が特異です")
                break
            
            # 記録
            self.theta_history.append(np.rad2deg(theta))
            self.scale_history.append(s)
            self.error_history.append(error)
            self.J_theta_history.append(J_theta)
            self.J_s_history.append(J_s)
            self.delta_theta_history.append(np.rad2deg(delta_theta))
            self.delta_s_history.append(delta_s)
            
            # パラメータ更新
            theta -= delta_theta
            s -= delta_s
            
            print(f"反復 {iteration+1}: θ={np.rad2deg(theta):.4f}度, s={s:.6f}, "
                  f"誤差={error:.2f}, Δθ={np.rad2deg(delta_theta):.6f}, Δs={delta_s:.6f}")
            
            # 収束判定
            if abs(delta_theta) < threshold_theta and abs(delta_s) < threshold_scale:
                print(f"\n収束しました（反復回数: {iteration+1}）")
                print(f"収束基準: |Δθ| < {np.rad2deg(threshold_theta):.6f}度, |Δs| < {threshold_scale:.6f}")
                break
                
        if iteration == max_iter - 1:
            print(f"\n最大反復回数 {max_iter} に到達しました")
            print(f"最終更新量: Δθ={np.rad2deg(delta_theta):.6f}度, Δs={delta_s:.6f}")
            print(f"収束基準: |Δθ| < {np.rad2deg(threshold_theta):.6f}度, |Δs| < {threshold_scale:.6f}")
        
        self.final_theta = theta
        self.final_s = s
        
        return theta, s
    
    def save_convergence_plots(self, save_dir="results"):
        """収束過程のグラフを保存"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        iterations = range(len(self.theta_history))
        
        # 1. パラメータの推定過程
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(iterations, self.theta_history, 'b-', linewidth=2)
        ax1.set_xlabel('反復回数')
        ax1.set_ylabel('回転角度 θ [度]')
        ax1.set_title('回転角度の収束過程')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(iterations, self.scale_history, 'r-', linewidth=2)
        ax2.set_xlabel('反復回数')
        ax2.set_ylabel('スケール s')
        ax2.set_title('スケールの収束過程')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parameter_convergence.png'), dpi=150)
        plt.close()
        
        # 2. 1階微分（勾配）の変化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(iterations, self.J_theta_history, 'b-', linewidth=2)
        ax1.set_xlabel('反復回数')
        ax1.set_ylabel('∂J/∂θ')
        ax1.set_title('目的関数のθに関する1階微分')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax2.plot(iterations, self.J_s_history, 'r-', linewidth=2)
        ax2.set_xlabel('反復回数')
        ax2.set_ylabel('∂J/∂s')
        ax2.set_title('目的関数のsに関する1階微分')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gradient_convergence.png'), dpi=150)
        plt.close()
        
        # 3. パラメータ更新量の変化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.semilogy(iterations, np.abs(self.delta_theta_history), 'b-', linewidth=2)
        ax1.set_xlabel('反復回数')
        ax1.set_ylabel('|Δθ| [度]（対数スケール）')
        ax1.set_title('回転角度の更新量')
        ax1.grid(True, alpha=0.3)
        
        ax2.semilogy(iterations, np.abs(self.delta_s_history), 'r-', linewidth=2)
        ax2.set_xlabel('反復回数')
        ax2.set_ylabel('|Δs|（対数スケール）')
        ax2.set_title('スケールの更新量')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'delta_convergence.png'), dpi=150)
        plt.close()
        
        # 4. 誤差の変化
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, self.error_history, 'g-', linewidth=2)
        ax.set_xlabel('反復回数')
        ax.set_ylabel('目的関数 J')
        ax.set_title('目的関数（誤差）の収束過程')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_convergence.png'), dpi=150)
        plt.close()
        
        print(f"\nグラフを {save_dir} ディレクトリに保存しました")

def main():
    parser = argparse.ArgumentParser(description='ガウス・ニュートン法による相似変換パラメータ推定')
    parser.add_argument('-i', '--input', required=True, help='入力画像ファイル名')
    parser.add_argument('-o', '--output', required=True, help='出力画像ファイル名')
    parser.add_argument('--init_theta', type=float, default=0, help='初期回転角度（度）')
    parser.add_argument('--init_scale', type=float, default=1.0, help='初期スケール')
    parser.add_argument('--max_iter', type=int, default=100, help='最大反復回数')
    parser.add_argument('--threshold', type=float, default=1e-5, help='収束判定閾値')
    parser.add_argument('--threshold_theta', type=float, default=None, help='回転角度の収束判定閾値（度）')
    parser.add_argument('--threshold_scale', type=float, default=None, help='スケールの収束判定閾値')
    parser.add_argument('--save_dir', default='results', help='結果保存ディレクトリ')
    parser.add_argument('--no_display', action='store_true', help='画像を表示しない')
    
    args = parser.parse_args()
    
    # 画像の読み込み
    img_in = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img_out = cv2.imread(args.output, cv2.IMREAD_GRAYSCALE)
    
    if img_in is None or img_out is None:
        print("エラー: 画像を読み込めません")
        return
    
    print(f"入力画像: {args.input} (サイズ: {img_in.shape})")
    print(f"出力画像: {args.output} (サイズ: {img_out.shape})")
    
    # 推定器の作成
    estimator = GaussNewtonEstimator(img_in, img_out)
    
    # パラメータ推定
    theta_rad = np.deg2rad(args.init_theta)
    
    # 個別の閾値設定
    threshold_theta = args.threshold_theta
    threshold_scale = args.threshold_scale
    
    theta_est, s_est = estimator.estimate(
        init_theta=theta_rad,
        init_s=args.init_scale,
        max_iter=args.max_iter,
        threshold=args.threshold,
        threshold_theta=threshold_theta,
        threshold_scale=threshold_scale
    )
    
    print(f"\n推定結果:")
    print(f"  回転角度: {np.rad2deg(theta_est):.4f} 度")
    print(f"  スケール: {s_est:.6f}")
    
    # グラフの保存
    estimator.save_convergence_plots(args.save_dir)

if __name__ == '__main__':
    main()
    