"""
ガウス・ニュートン法による相似変換パラメータ推定プログラム
数値解析・最適化工学特論 課題2

【概要】
入力画像と相似変換した出力画像から回転角度θとスケールパラメータsを
ガウス・ニュートン法によって推定する

【使用方法】
python gauss_newton_estimation.py input_image.jpg output_image.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class SimilarityTransformEstimator:
    def __init__(self, gaussian_kernel_size=7, gaussian_sigma=2.0):
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        
    def create_smooth_derivative_images(self, image):
        """
        平滑微分画像を作成する
        ガウシアンフィルタでノイズを低減後、x方向・y方向の微分を計算
        """
        # ガウシアンフィルタによる平滑化
        blurred = cv2.GaussianBlur(image, (self.gaussian_kernel_size, self.gaussian_kernel_size), 
                                 sigmaX=self.gaussian_sigma, sigmaY=self.gaussian_sigma)
        
        # 微分フィルタ（Sobel演算子）
        dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        return dx, dy
    
    def apply_similarity_transform(self, image, theta, scale):
        """
        画像に相似変換を適用する（逆変換）
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # OpenCVの回転行列を使用（より安定）
        rotation_matrix = cv2.getRotationMatrix2D(center, np.rad2deg(theta), scale)
        
        # 相似変換の適用
        transformed = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=0)
        
        return transformed
    
    def crop_to_circle(self, image):
        """
        画像を円形にクロップする
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center)
        
        # 円形マスクの作成
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # マスクの適用
        if len(image.shape) == 3:
            # カラー画像の場合：各チャンネルに同じマスクを適用
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = cv2.bitwise_and(image[:, :, i], mask)
        else:
            # グレースケール画像の場合
            result = cv2.bitwise_and(image, mask)
            
        return result
    
    def estimate_parameters(self, input_image, output_image, theta_init=0.0, scale_init=1.0, 
                          max_iterations=100, threshold=1e-6):
        """
        ガウス・ニュートン法によるパラメータ推定
        """
        # 画像サイズが大きい場合はリサイズ
        h, w = input_image.shape[:2]
        if max(h, w) > 512:
            scale_factor = 512 / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            input_image = cv2.resize(input_image, (new_w, new_h))
            output_image = cv2.resize(output_image, (new_w, new_h))
            print(f"画像をリサイズ: {w}x{h} → {new_w}x{new_h}")
        
        # 初期値設定
        theta = np.deg2rad(theta_init)
        scale = scale_init
        
        # 履歴保存用
        theta_history = [np.rad2deg(theta)]
        scale_history = [scale]
        objective_history = []
        
        # 画像の前処理（正規化）
        input_img = input_image.astype(np.float64) / 255.0
        output_img = output_image.astype(np.float64) / 255.0
        
        h, w = input_img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 円形マスクの作成
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        x_centered = x_coords - center_x
        y_centered = y_coords - center_y
        radius = min(center_x, center_y) - 10
        mask = (x_centered**2 + y_centered**2) <= radius**2
        
        print(f"初期値: θ = {np.rad2deg(theta):.2f}°, s = {scale:.4f}")
        print(f"有効ピクセル数: {np.sum(mask)}")
        
        for iteration in range(max_iterations):
            # 現在のパラメータで出力画像を変換（入力画像に合わせる）
            cos_theta = np.cos(-theta)  # 逆回転
            sin_theta = np.sin(-theta)
            inv_scale = 1.0 / scale     # 逆スケール
            
            # 変換された出力画像
            transformed_output = np.zeros_like(output_img)
            
            # 有効な領域でのみ計算（ベクトル化）
            valid_indices = np.where(mask)
            
            for i in range(len(valid_indices[0])):
                y, x = valid_indices[0][i], valid_indices[1][i]
                
                # 座標変換（画像中心基準）
                x_rel = x - center_x
                y_rel = y - center_y
                
                # 相似変換の逆変換適用
                x_src = inv_scale * (cos_theta * x_rel - sin_theta * y_rel) + center_x
                y_src = inv_scale * (sin_theta * x_rel + cos_theta * y_rel) + center_y
                
                # 最近傍補間
                x_src_int = int(round(x_src))
                y_src_int = int(round(y_src))
                
                if 0 <= x_src_int < w and 0 <= y_src_int < h:
                    transformed_output[y, x] = output_img[y_src_int, x_src_int]
            
            # 平滑微分画像の作成
            I_x_prime, I_y_prime = self.create_smooth_derivative_images(transformed_output)
            
            # 画像差分（マスク内のみ）
            image_diff = np.zeros_like(input_img)
            image_diff[mask] = input_img[mask] - transformed_output[mask]
            
            # 偏微分の計算（正しい符号で）
            dx_prime_dtheta = -scale * (-x_centered * sin_theta - y_centered * cos_theta)
            dy_prime_dtheta = -scale * (x_centered * cos_theta - y_centered * sin_theta)
            
            dx_prime_dscale = -(x_centered * cos_theta - y_centered * sin_theta)
            dy_prime_dscale = -(x_centered * sin_theta + y_centered * cos_theta)
            
            # 1階微分の計算
            grad_theta_term = I_x_prime * dx_prime_dtheta + I_y_prime * dy_prime_dtheta
            grad_scale_term = I_x_prime * dx_prime_dscale + I_y_prime * dy_prime_dscale
            
            J_theta = np.sum(image_diff[mask] * grad_theta_term[mask])
            J_scale = np.sum(image_diff[mask] * grad_scale_term[mask])
            
            # 2階微分の計算（ガウス・ニュートン近似）
            J_theta_theta = np.sum((grad_theta_term[mask]) ** 2)
            J_scale_scale = np.sum((grad_scale_term[mask]) ** 2)
            J_theta_scale = np.sum(grad_theta_term[mask] * grad_scale_term[mask])
            
            # ヘッセ行列と勾配ベクトル
            H = np.array([[J_theta_theta, J_theta_scale],
                         [J_theta_scale, J_scale_scale]])
            
            grad = np.array([J_theta, J_scale])
            
            # 条件数をチェック
            if np.linalg.cond(H) > 1e12:
                print(f"ヘッセ行列の条件数が悪い: {np.linalg.cond(H):.2e}")
                break
            
            # パラメータ更新量の計算
            try:
                delta = np.linalg.solve(H, -grad)
                delta_theta, delta_scale = delta
            except np.linalg.LinAlgError:
                print(f"特異行列により反復{iteration}で終了")
                break
            
            # 更新量を制限（発散防止）
            max_theta_step = np.deg2rad(5)   # 5度まで
            max_scale_step = 0.05            # 0.05まで
            
            if abs(delta_theta) > max_theta_step:
                delta_theta = np.sign(delta_theta) * max_theta_step
            if abs(delta_scale) > max_scale_step:
                delta_scale = np.sign(delta_scale) * max_scale_step
            
            # パラメータの更新
            theta += delta_theta
            scale += delta_scale
            
            # スケールの範囲制限
            scale = max(0.1, min(3.0, scale))
            
            # 目的関数値の計算
            objective_value = 0.5 * np.sum(image_diff[mask] ** 2)
            
            # 履歴の保存
            theta_history.append(np.rad2deg(theta))
            scale_history.append(scale)
            objective_history.append(objective_value)
            
            print(f"反復 {iteration+1}: θ = {np.rad2deg(theta):6.2f}°, s = {scale:6.4f}, "
                  f"Δθ = {np.rad2deg(delta_theta):8.4f}°, Δs = {delta_scale:8.4f}, "
                  f"J = {objective_value:.2e}")
            
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
    
    def visualize_results(self, results, true_theta=None, true_scale=None, save_plots=True):
        """
        結果の可視化
        """
        # matplotlibのバックエンドを設定（GUI不要）
        import matplotlib
        matplotlib.use('Agg')  # GUI不要のバックエンド
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # θの収束過程
        axes[0, 0].plot(results['theta_history'], 'b-', linewidth=2, label='推定値')
        if true_theta is not None:
            axes[0, 0].axhline(y=true_theta, color='r', linestyle='--', linewidth=2, label=f'真値 ({true_theta}°)')
        axes[0, 0].set_xlabel('反復回数')
        axes[0, 0].set_ylabel('回転角度 θ (度)')
        axes[0, 0].set_title('回転角度の収束過程')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # sの収束過程
        axes[0, 1].plot(results['scale_history'], 'g-', linewidth=2, label='推定値')
        if true_scale is not None:
            axes[0, 1].axhline(y=true_scale, color='r', linestyle='--', linewidth=2, label=f'真値 ({true_scale})')
        axes[0, 1].set_xlabel('反復回数')
        axes[0, 1].set_ylabel('スケールパラメータ s')
        axes[0, 1].set_title('スケールパラメータの収束過程')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # 目的関数の変化
        if len(results['objective_history']) > 0:
            axes[1, 0].semilogy(results['objective_history'], 'purple', linewidth=2)
            axes[1, 0].set_xlabel('反復回数')
            axes[1, 0].set_ylabel('目的関数 J')
            axes[1, 0].set_title('目的関数の変化')
            axes[1, 0].grid(True)
        
        # パラメータ空間での軌跡
        axes[1, 1].plot(results['theta_history'], results['scale_history'], 'o-', linewidth=2, markersize=4)
        axes[1, 1].plot(results['theta_history'][0], results['scale_history'][0], 'go', markersize=8, label='初期値')
        axes[1, 1].plot(results['theta_history'][-1], results['scale_history'][-1], 'ro', markersize=8, label='最終値')
        if true_theta is not None and true_scale is not None:
            axes[1, 1].plot(true_theta, true_scale, 'r*', markersize=12, label='真値')
        axes[1, 1].set_xlabel('回転角度 θ (度)')
        axes[1, 1].set_ylabel('スケールパラメータ s')
        axes[1, 1].set_title('パラメータ空間での収束軌跡')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('convergence_results.png', dpi=300, bbox_inches='tight')
            print("収束結果をconvergence_results.pngに保存しました")
        
        # ファイルに保存するのみ、表示はしない
        plt.close()


def main():
    if len(sys.argv) != 3:
        print("使用方法: python gauss_newton_estimation.py input_image.jpg output_image.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 画像の読み込み
    input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    output_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    
    if input_image is None or output_image is None:
        print("エラー: 画像を読み込めませんでした")
        sys.exit(1)
    
    print(f"入力画像: {input_path} ({input_image.shape})")
    print(f"出力画像: {output_path} ({output_image.shape})")
    
    # 推定器の作成
    estimator = SimilarityTransformEstimator(gaussian_kernel_size=7, gaussian_sigma=2.0)
    
    # 円形にクロップ
    input_cropped = estimator.crop_to_circle(input_image)
    output_cropped = estimator.crop_to_circle(output_image)
    
    # 画像の表示
    cv2.imshow('Input Image', input_cropped)
    cv2.imshow('Output Image', output_cropped)
    cv2.waitKey(1)
    
    # パラメータ推定
    print("\n=== ガウス・ニュートン法による推定開始 ===")
    results = estimator.estimate_parameters(
        input_cropped, output_cropped,
        theta_init=0.0,  # 初期値: 0度
        scale_init=1.0,  # 初期値: 1.0
        max_iterations=100,
        threshold=1e-6
    )
    
    # 結果の表示
    print(f"\n=== 推定結果 ===")
    print(f"推定された回転角度: {results['theta']:.4f}°")
    print(f"推定されたスケールパラメータ: {results['scale']:.4f}")
    print(f"収束反復回数: {results['iterations']}")
    
    # 結果の可視化
    # 真値がわかっている場合は以下のように指定
    # estimator.visualize_results(results, true_theta=45.0, true_scale=0.5)
    estimator.visualize_results(results)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()