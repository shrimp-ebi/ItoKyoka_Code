import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
import os

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

class SimilarityTransformEstimator:
    def __init__(self, input_image_path, true_theta, true_s):
        """
        相似変換パラメータ推定クラス
        
        Args:
            input_image_path: 入力画像のパス
            true_theta: 真の回転角度（ラジアン）
            true_s: 真のスケールパラメータ
        """
        self.input_image = self.load_image(input_image_path)
        self.true_theta = true_theta
        self.true_s = true_s
        self.height, self.width = self.input_image.shape
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.radius = min(self.center_x, self.center_y) - 10  # 境界マージン
        
        # 出力画像を生成
        self.output_image = self.apply_similarity_transform(self.input_image, true_theta, true_s)
        
        # 平滑微分画像を作成
        self.output_grad_x, self.output_grad_y = self.compute_smooth_gradients(self.output_image)
        
    def load_image(self, image_path):
        """画像を読み込み、グレースケールに変換"""
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('L')
        else:
            # shrimp.pngが存在しない場合、サンプル画像を生成
            print(f"Warning: {image_path} not found. Creating sample image.")
            img = self.create_sample_image()
        
        return np.array(img, dtype=np.float64)
    
    def create_sample_image(self):
        """サンプル画像を生成（エビのような模様）"""
        size = 256
        img = np.zeros((size, size))
        
        # 楕円形状
        y, x = np.ogrid[:size, :size]
        center_x, center_y = size//2, size//2
        
        # メイン楕円
        ellipse1 = ((x - center_x)**2 / 80**2 + (y - center_y)**2 / 40**2) <= 1
        img[ellipse1] = 180
        
        # 頭部円
        head = ((x - center_x + 60)**2 + (y - center_y)**2) <= 25**2
        img[head] = 200
        
        # 尻尾
        tail = ((x - center_x - 80)**2 / 15**2 + (y - center_y)**2 / 8**2) <= 1
        img[tail] = 160
        
        # 縞模様
        for i in range(5):
            stripe_x = center_x - 40 + i * 15
            stripe = np.abs(x - stripe_x) <= 3
            img[stripe & ellipse1] = 120
            
        return Image.fromarray(img.astype(np.uint8))
    
    def apply_similarity_transform(self, image, theta, s):
        """相似変換を適用"""
        height, width = image.shape
        output = np.zeros_like(image)
        
        # 変換行列
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        for v in range(height):
            for u in range(width):
                # 画像座標から数学座標への変換
                x = u - self.center_x
                y = v - self.center_y
                
                # 円形クロップ
                if x*x + y*y > self.radius*self.radius:
                    continue
                
                # 逆変換（出力座標から入力座標を求める）
                x_orig = (x * cos_theta + y * sin_theta) / s
                y_orig = (-x * sin_theta + y * cos_theta) / s
                
                # 数学座標から画像座標への変換
                u_orig = x_orig + self.center_x
                v_orig = y_orig + self.center_y
                
                # 最近傍補間
                u_orig_int = int(round(u_orig))
                v_orig_int = int(round(v_orig))
                
                if (0 <= u_orig_int < width and 0 <= v_orig_int < height):
                    output[v, u] = image[v_orig_int, u_orig_int]
        
        return output
    
    def compute_smooth_gradients(self, image, sigma=2):
        """平滑微分画像を計算"""
        # ガウシアンフィルタで平滑化
        smoothed = gaussian_filter(image, sigma=sigma)
        
        # 差分フィルタで微分
        grad_x = np.zeros_like(smoothed)
        grad_y = np.zeros_like(smoothed)
        
        # x方向微分
        grad_x[:, 1:-1] = (smoothed[:, 2:] - smoothed[:, :-2]) / 2.0
        
        # y方向微分
        grad_y[1:-1, :] = (smoothed[2:, :] - smoothed[:-2, :]) / 2.0
        
        return grad_x, grad_y
    
    def compute_objective_function(self, theta, s):
        """目的関数Jを計算"""
        transformed = self.apply_similarity_transform(self.input_image, theta, s)
        diff = transformed - self.output_image
        
        # 円形領域内のみ計算
        y, x = np.ogrid[:self.height, :self.width]
        x_centered = x - self.center_x
        y_centered = y - self.center_y
        mask = (x_centered**2 + y_centered**2) <= self.radius**2
        
        return 0.5 * np.sum(diff[mask]**2)
    
    def compute_derivatives(self, theta, s):
        """1階・2階微分を計算"""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 変換画像を生成
        transformed = self.apply_similarity_transform(self.input_image, theta, s)
        diff = transformed - self.output_image
        
        # 微分項の初期化
        J_theta = 0.0
        J_s = 0.0
        J_theta_theta = 0.0
        J_s_s = 0.0
        J_theta_s = 0.0
        
        for v in range(self.height):
            for u in range(self.width):
                # 画像座標から数学座標
                x = u - self.center_x
                y = v - self.center_y
                
                # 円形クロップ
                if x*x + y*y > self.radius*self.radius:
                    continue
                
                # 偏微分項の計算
                dx_dtheta = -s * (x * sin_theta + y * cos_theta)
                dy_dtheta = s * (x * cos_theta - y * sin_theta)
                dx_ds = x * cos_theta - y * sin_theta
                dy_ds = x * sin_theta + y * cos_theta
                
                # 変換後座標での勾配
                grad_term_theta = (self.output_grad_x[v, u] * dx_dtheta + 
                                 self.output_grad_y[v, u] * dy_dtheta)
                grad_term_s = (self.output_grad_x[v, u] * dx_ds + 
                              self.output_grad_y[v, u] * dy_ds)
                
                # 1階微分
                J_theta += diff[v, u] * grad_term_theta
                J_s += diff[v, u] * grad_term_s
                
                # 2階微分（ガウス・ニュートン近似）
                J_theta_theta += grad_term_theta**2
                J_s_s += grad_term_s**2
                J_theta_s += grad_term_theta * grad_term_s
        
        return J_theta, J_s, J_theta_theta, J_s_s, J_theta_s
    
    def gauss_newton_estimation(self, init_theta, init_s, max_iter=50, tol=1e-6):
        """ガウス・ニュートン法によるパラメータ推定"""
        theta = init_theta
        s = init_s
        
        theta_history = [theta]
        s_history = [s]
        j_history = []
        
        print(f"初期値: theta = {np.degrees(theta):.2f}°, s = {s:.3f}")
        print(f"真値: theta = {np.degrees(self.true_theta):.2f}°, s = {self.true_s:.3f}")
        print("-" * 50)
        
        for iteration in range(max_iter):
            # 目的関数値
            J = self.compute_objective_function(theta, s)
            j_history.append(J)
            
            # 1階・2階微分計算
            J_theta, J_s, J_theta_theta, J_s_s, J_theta_s = self.compute_derivatives(theta, s)
            
            # ヘッセ行列
            H = np.array([[J_theta_theta, J_theta_s],
                         [J_theta_s, J_s_s]])
            
            # 勾配ベクトル
            g = np.array([J_theta, J_s])
            
            # パラメータ更新量の計算
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                print(f"反復 {iteration}: ヘッセ行列が特異です")
                break
            
            # パラメータ更新
            theta -= delta[0]
            s -= delta[1]
            
            theta_history.append(theta)
            s_history.append(s)
            
            print(f"反復 {iteration+1}: theta = {np.degrees(theta):.4f}°, s = {s:.6f}, J = {J:.2f}")
            
            # 収束判定
            if np.abs(delta[0]) < tol and np.abs(delta[1]) < tol:
                print(f"\n収束しました（反復回数: {iteration+1}）")
                break
        
        return theta, s, theta_history, s_history, j_history
    
    def create_objective_function_surface(self, theta_range, s_range, resolution=50):
        """目的関数の等高線図用データを作成"""
        theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)
        s_vals = np.linspace(s_range[0], s_range[1], resolution)
        
        J_vals = np.zeros((len(s_vals), len(theta_vals)))
        
        for i, s_val in enumerate(s_vals):
            for j, theta_val in enumerate(theta_vals):
                J_vals[i, j] = self.compute_objective_function(theta_val, s_val)
        
        return theta_vals, s_vals, J_vals
    
    def visualize_results(self, theta_est, s_est, theta_history, s_history, j_history):
        """結果の可視化"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 入力画像と出力画像
        ax1 = plt.subplot(2, 4, 1)
        plt.imshow(self.input_image, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        
        ax2 = plt.subplot(2, 4, 2)
        plt.imshow(self.output_image, cmap='gray')
        plt.title(f'Output Image\n(θ={np.degrees(self.true_theta):.1f}°, s={self.true_s:.2f})')
        plt.axis('off')
        
        # 2. 推定結果画像
        estimated_image = self.apply_similarity_transform(self.input_image, theta_est, s_est)
        ax3 = plt.subplot(2, 4, 3)
        plt.imshow(estimated_image, cmap='gray')
        plt.title(f'Estimated Transform\n(θ={np.degrees(theta_est):.1f}°, s={s_est:.3f})')
        plt.axis('off')
        
        # 3. 差分画像
        ax4 = plt.subplot(2, 4, 4)
        diff = np.abs(estimated_image - self.output_image)
        plt.imshow(diff, cmap='hot')
        plt.title('Absolute Difference')
        plt.colorbar()
        plt.axis('off')
        
        # 4. 目的関数の等高線図と収束軌跡
        ax5 = plt.subplot(2, 4, 5)
        theta_range = [self.true_theta - 0.5, self.true_theta + 0.5]
        s_range = [self.true_s - 0.3, self.true_s + 0.3]
        
        theta_vals, s_vals, J_vals = self.create_objective_function_surface(
            theta_range, s_range, resolution=30)
        
        plt.contour(np.degrees(theta_vals), s_vals, J_vals, levels=20)
        plt.plot(np.degrees(theta_history), s_history, 'ro-', markersize=4, linewidth=2, 
                label='Convergence Path')
        plt.plot(np.degrees(self.true_theta), self.true_s, 'g*', markersize=15, label='True Value')
        plt.plot(np.degrees(theta_est), s_est, 'b^', markersize=10, label='Estimated')
        plt.xlabel('θ (degrees)')
        plt.ylabel('s')
        plt.title('Objective Function Contour')
        plt.legend()
        plt.grid(True)
        
        # 5. 収束過程（目的関数値）
        ax6 = plt.subplot(2, 4, 6)
        plt.semilogy(j_history, 'b-o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function J')
        plt.title('Convergence Process')
        plt.grid(True)
        
        # 6. θ方向の断面
        ax7 = plt.subplot(2, 4, 7)
        theta_slice = np.linspace(theta_range[0], theta_range[1], 100)
        j_theta_slice = [self.compute_objective_function(t, self.true_s) for t in theta_slice]
        plt.plot(np.degrees(theta_slice), j_theta_slice, 'b-', linewidth=2)
        plt.axvline(np.degrees(self.true_theta), color='g', linestyle='--', label='True θ')
        plt.axvline(np.degrees(theta_est), color='r', linestyle='--', label='Estimated θ')
        plt.xlabel('θ (degrees)')
        plt.ylabel('J')
        plt.title('J vs θ (s fixed at true value)')
        plt.legend()
        plt.grid(True)
        
        # 7. s方向の断面
        ax8 = plt.subplot(2, 4, 8)
        s_slice = np.linspace(s_range[0], s_range[1], 100)
        j_s_slice = [self.compute_objective_function(self.true_theta, s) for s in s_slice]
        plt.plot(s_slice, j_s_slice, 'b-', linewidth=2)
        plt.axvline(self.true_s, color='g', linestyle='--', label='True s')
        plt.axvline(s_est, color='r', linestyle='--', label='Estimated s')
        plt.xlabel('s')
        plt.ylabel('J')
        plt.title('J vs s (θ fixed at true value)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 結果の考察を表示
        self.print_analysis(theta_est, s_est, len(j_history))
    
    def print_analysis(self, theta_est, s_est, iterations):
        """結果の考察を出力"""
        theta_error = np.degrees(abs(theta_est - self.true_theta))
        s_error = abs(s_est - self.true_s)
        
        print("\n" + "="*60)
        print("結果の考察")
        print("="*60)
        print(f"収束反復回数: {iterations}")
        print(f"真値:     θ = {np.degrees(self.true_theta):.4f}°, s = {self.true_s:.6f}")
        print(f"推定値:   θ = {np.degrees(theta_est):.4f}°, s = {s_est:.6f}")
        print(f"誤差:     Δθ = {theta_error:.4f}°, Δs = {s_error:.6f}")
        print(f"相対誤差: θ: {theta_error/abs(np.degrees(self.true_theta))*100:.3f}%, "
              f"s: {s_error/abs(self.true_s)*100:.3f}%")
        
        print("\n考察:")
        if theta_error < 1.0 and s_error < 0.01:
            print("・高精度での推定が達成されました")
        elif theta_error < 5.0 and s_error < 0.05:
            print("・良好な推定精度が得られました")
        else:
            print("・推定精度に改善の余地があります")
            
        if iterations < 10:
            print("・高速収束が確認されました")
        elif iterations < 20:
            print("・適度な収束速度でした")
        else:
            print("・収束に時間を要しました。初期値や画像特徴の影響が考えられます")

def main():
    """メイン実行関数"""
    # パラメータ設定
    true_theta = np.radians(15.0)  # 15度回転
    true_s = 1.2                   # 1.2倍拡大
    
    # 推定器の初期化
    estimator = SimilarityTransformEstimator('shrimp.png', true_theta, true_s)
    
    # 初期値設定（真値から少しずらす）
    init_theta = np.radians(10.0)  # 10度から開始
    init_s = 1.0                   # 1.0から開始
    
    # ガウス・ニュートン法実行
    theta_est, s_est, theta_history, s_history, j_history = estimator.gauss_newton_estimation(
        init_theta, init_s, max_iter=50, tol=1e-6)
    
    # 結果の可視化
    estimator.visualize_results(theta_est, s_est, theta_history, s_history, j_history)

if __name__ == "__main__":
    main()