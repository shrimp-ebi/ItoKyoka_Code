"""
ガウス・ニュートン法による相似変換パラメータ推定プログラム
入力画像と出力画像から回転角度θとスケールsを推定します。

使用方法:
python gauss.py --input Fuji.jpg --output out-Fuji-gray.png --theta_init 10 --scale_init 1.2

初期値を変える場合：
python gauss.py --input Fuji.jpg --output out-Fuji-gray.png --theta_init 10 --scale_init 1.2
"""

"""
ガウス・ニュートン法による相似変換パラメータ推定プログラム
入力画像と出力画像から回転角度θとスケールsを推定します。

使用方法:
python gauss.py --input input.jpg --output output.png --theta_init 0 --scale_init 1
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # GUIを使わないバックエンドを指定
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

class GaussNewtonEstimator:
    def __init__(self, img_in, img_out, theta_init=0, scale_init=1):
        """
        Parameters:
        -----------
        img_in : numpy.ndarray
            入力画像（グレースケール）
        img_out : numpy.ndarray
            出力画像（グレースケール）
        theta_init : float
            回転角度の初期値（度数法）
        scale_init : float
            スケールの初期値
        """
        self.img_in = img_in.astype(np.float64)
        self.img_out = img_out.astype(np.float64)
        self.h, self.w = img_in.shape
        
        # 画像中心
        self.cx = self.w / 2
        self.cy = self.h / 2
        
        # パラメータ初期化
        self.theta = np.deg2rad(theta_init)
        self.s = scale_init
        
        # 収束履歴
        self.theta_history = []
        self.s_history = []
        self.error_history = []
        self.J_theta_history = []
        self.J_s_history = []
        
        # 有効ピクセルのマスクと座標を事前計算
        self._prepare_valid_pixels()
        
        # ガウシアンフィルタの準備
        self._prepare_gaussian_filters()
    
    def _prepare_valid_pixels(self):
        """画像中心から半径内の有効ピクセルを計算"""
        radius = min(self.w, self.h) / 2
        y, x = np.meshgrid(range(self.h), range(self.w), indexing='ij')
        
        # 中心からの距離
        dist = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        
        # 有効ピクセルのマスク
        self.valid_mask = dist <= radius
        
        # 有効ピクセルの座標
        self.valid_pixels = np.column_stack(np.where(self.valid_mask))
        self.n_valid = len(self.valid_pixels)
    
    def _prepare_gaussian_filters(self, sigma=1.0):
        """平滑微分用のガウシアンフィルタを準備"""
        size = int(1 + 4 * sigma)
        if size % 2 == 0:
            size += 1
        
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        X, Y = np.meshgrid(x, y)
        
        # ガウシアン関数
        G = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        G = G / (2 * np.pi * sigma**2)
        
        # x方向、y方向の微分
        self.Gx = -X / sigma**2 * G
        self.Gy = -Y / sigma**2 * G
    
    def _compute_derivatives(self):
        """画像の平滑微分を計算"""
        # 出力画像の平滑微分
        self.Ix = cv2.filter2D(self.img_out, -1, self.Gx)
        self.Iy = cv2.filter2D(self.img_out, -1, self.Gy)
    
    def _transform_point(self, x, y):
        """点(x,y)を現在のパラメータで変換（Yu-gauss.pyと同じ座標系）"""
        # 中心を原点に
        x_c = x - self.cx
        y_c = self.cy - y  # Y座標を反転（OpenCV座標系）
        
        # 相似変換
        x_new = self.s * (x_c * np.cos(self.theta) - y_c * np.sin(self.theta))
        y_new = -self.s * (x_c * np.sin(self.theta) + y_c * np.cos(self.theta))  # 負の符号
        
        # 元の座標系に戻す
        return x_new + self.cx, y_new + self.cy
    
    def _compute_jacobian(self):
        """目的関数の1階・2階微分を計算"""
        J_theta = 0
        J_s = 0
        J_theta_theta = 0
        J_s_s = 0
        J_theta_s = 0
        error = 0
        
        for i in range(self.n_valid):
            y, x = self.valid_pixels[i]
            
            # 変換後の座標
            x_t, y_t = self._transform_point(x, y)
            
            # 画像範囲内かチェック
            if 0 <= x_t < self.w and 0 <= y_t < self.h:
                # バイリニア補間
                x_int = int(x_t)
                y_int = int(y_t)
                dx = x_t - x_int
                dy = y_t - y_int
                
                if x_int + 1 < self.w and y_int + 1 < self.h:
                    # 画素値の補間
                    I_out = (1-dx)*(1-dy)*self.img_out[y_int, x_int] + \
                            dx*(1-dy)*self.img_out[y_int, x_int+1] + \
                            (1-dx)*dy*self.img_out[y_int+1, x_int] + \
                            dx*dy*self.img_out[y_int+1, x_int+1]
                    
                    # 微分値の補間
                    Ix_val = (1-dx)*(1-dy)*self.Ix[y_int, x_int] + \
                             dx*(1-dy)*self.Ix[y_int, x_int+1] + \
                             (1-dx)*dy*self.Ix[y_int+1, x_int] + \
                             dx*dy*self.Ix[y_int+1, x_int+1]
                    
                    Iy_val = (1-dx)*(1-dy)*self.Iy[y_int, x_int] + \
                             dx*(1-dy)*self.Iy[y_int, x_int+1] + \
                             (1-dx)*dy*self.Iy[y_int+1, x_int] + \
                             dx*dy*self.Iy[y_int+1, x_int+1]
                else:
                    continue
            else:
                continue
            
            # 入力画像の画素値
            I_in = self.img_in[y, x]
            
            # 誤差
            diff = I_out - I_in
            
            # 座標の微分（Yu-gauss.pyと同じ座標系）
            x_c = x - self.cx
            y_c = self.cy - y  # Y座標を反転
            
            # θに関する微分
            dx_dtheta = self.s * (-(x_c) * np.sin(self.theta) - y_c * np.cos(self.theta))
            dy_dtheta = self.s * (x_c * np.cos(self.theta) - y_c * np.sin(self.theta))
            
            # sに関する微分
            dx_ds = x_c * np.cos(self.theta) - y_c * np.sin(self.theta)
            dy_ds = x_c * np.sin(self.theta) + y_c * np.cos(self.theta)
            
            # 1階微分
            dI_dtheta = Ix_val * dx_dtheta + Iy_val * dy_dtheta
            dI_ds = Ix_val * dx_ds + Iy_val * dy_ds
            
            J_theta += diff * dI_dtheta
            J_s += diff * dI_ds
            
            # 2階微分（ガウス・ニュートン近似）
            J_theta_theta += dI_dtheta * dI_dtheta
            J_s_s += dI_ds * dI_ds
            J_theta_s += dI_dtheta * dI_ds
            
            # 誤差の累積
            error += 0.5 * diff * diff
        
        return J_theta, J_s, J_theta_theta, J_s_s, J_theta_s, error
    
    def optimize(self, max_iter=100, threshold=1e-5):
        """ガウス・ニュートン法による最適化"""
        print(f"初期値: θ={np.rad2deg(self.theta):.2f}°, s={self.s:.4f}")
        
        # 平滑微分の計算
        self._compute_derivatives()
        
        for i in range(max_iter):
            # ヤコビアンの計算
            J_theta, J_s, J_theta_theta, J_s_s, J_theta_s, error = self._compute_jacobian()
            
            # 履歴の保存
            self.theta_history.append(np.rad2deg(self.theta))
            self.s_history.append(self.s)
            self.error_history.append(error)
            self.J_theta_history.append(J_theta)
            self.J_s_history.append(J_s)
            
            # ヘッセ行列とグラディエントベクトル
            H = np.array([[J_theta_theta, J_theta_s],
                          [J_theta_s, J_s_s]])
            g = np.array([J_theta, J_s])
            
            # パラメータ更新量の計算
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                print("ヘッセ行列が特異です")
                break
            
            delta_theta = delta[0]
            delta_s = delta[1]
            
            # 収束判定
            if abs(delta_theta) < threshold and abs(delta_s) < threshold:
                print(f"収束しました（反復回数: {i+1}）")
                break
            
            # パラメータ更新
            self.theta -= delta_theta
            self.s -= delta_s
            
            print(f"反復 {i+1}: θ={np.rad2deg(self.theta):.4f}°, s={self.s:.4f}, "
                  f"誤差={error:.2f}, |Δθ|={abs(delta_theta):.2e}, |Δs|={abs(delta_s):.2e}")
        
        print(f"\n推定結果: θ={np.rad2deg(self.theta):.4f}°, s={self.s:.4f}")
        
        return self.theta, self.s
    
    def plot_convergence(self, save_path='convergence_plots.png'):
        """収束過程のグラフを作成"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # θの推定過程
        ax = axes[0, 0]
        ax.plot(self.theta_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('θ [degrees]')
        ax.set_title('Rotation Angle Estimation')
        ax.grid(True)
        
        # sの推定過程
        ax = axes[0, 1]
        ax.plot(self.s_history, 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Scale')
        ax.set_title('Scale Parameter Estimation')
        ax.grid(True)
        
        # 誤差の推移
        ax = axes[0, 2]
        ax.semilogy(self.error_history, 'g-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        ax.set_title('Total Error')
        ax.grid(True)
        
        # θの1階微分
        ax = axes[1, 0]
        ax.plot(self.J_theta_history, 'b--', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('∂J/∂θ')
        ax.set_title('Gradient w.r.t. θ')
        ax.grid(True)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # sの1階微分
        ax = axes[1, 1]
        ax.plot(self.J_s_history, 'r--', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('∂J/∂s')
        ax.set_title('Gradient w.r.t. s')
        ax.grid(True)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # パラメータの変化（Δθ, Δs）
        ax = axes[1, 2]
        if len(self.theta_history) > 1:
            delta_theta = np.diff(self.theta_history)
            delta_s = np.diff(self.s_history)
            ax.semilogy(np.abs(delta_theta), 'b-', label='|Δθ|', linewidth=2)
            ax.semilogy(np.abs(delta_s), 'r-', label='|Δs|', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Change')
            ax.set_title('Convergence of Parameters')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()  # メモリ解放
        print(f"\n収束グラフを '{save_path}' に保存しました")

def main():
    parser = argparse.ArgumentParser(description='ガウス・ニュートン法による相似変換パラメータ推定')
    parser.add_argument('--input', '-i', type=str, default='in_shrimp.png',
                        help='入力画像ファイル名')
    parser.add_argument('--output', '-o', type=str, default='out_shrimp.png',
                        help='出力画像ファイル名')
    parser.add_argument('--theta_init', type=float, default=0,
                        help='回転角度の初期値（度数法）')
    parser.add_argument('--scale_init', type=float, default=1,
                        help='スケールの初期値')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='最大反復回数')
    
    args = parser.parse_args()
    
    # 画像の読み込み
    img_in = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img_out = cv2.imread(args.output, cv2.IMREAD_GRAYSCALE)
    
    if img_in is None or img_out is None:
        print("エラー: 画像を読み込めません")
        return
    
    print(f"入力画像: {args.input} (サイズ: {img_in.shape})")
    print(f"出力画像: {args.output} (サイズ: {img_out.shape})")
    
    # 推定器の初期化
    estimator = GaussNewtonEstimator(img_in, img_out, 
                                     theta_init=args.theta_init,
                                     scale_init=args.scale_init)
    
    # 最適化の実行
    theta_est, s_est = estimator.optimize(max_iter=args.max_iter)
    
    # 収束グラフの作成
    estimator.plot_convergence()
    
    # 結果をCSVに保存（オプション）
    import csv
    with open('estimation_history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'theta', 'scale', 'error', 'J_theta', 'J_s'])
        for i in range(len(estimator.theta_history)):
            writer.writerow([i, estimator.theta_history[i], estimator.s_history[i],
                           estimator.error_history[i], estimator.J_theta_history[i],
                           estimator.J_s_history[i]])
    print("\n推定履歴を 'estimation_history.csv' に保存しました")

if __name__ == '__main__':
    main()