"""
画像に相似変換を適用するプログラム
数値解析・最適化工学特論 課題2

【概要】
入力画像に指定された回転角度とスケールパラメータを適用して出力画像を生成する

【使用方法】
python apply_similarity_transform.py input_image.jpg theta_deg scale output_image.jpg

例：
python apply_similarity_transform.py lenna.jpg 45 0.5 lenna_transformed.jpg
"""

import cv2
import numpy as np
import sys

def apply_similarity_transform(image, theta_deg, scale):
    """
    画像に相似変換を適用する
    
    Args:
        image: 入力画像
        theta_deg: 回転角度（度）
        scale: スケールパラメータ
    
    Returns:
        transformed_image: 変換後の画像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 回転行列の作成（OpenCVの関数を使用）
    rotation_matrix = cv2.getRotationMatrix2D(center, theta_deg, scale)
    
    # 相似変換の適用
    transformed = cv2.warpAffine(image, rotation_matrix, (w, h), 
                               flags=cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=0)
    
    return transformed

def crop_to_circle(image):
    """
    画像を円形にクロップする
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center) - 10  # 境界から少し内側
    
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

def create_test_images(base_image_path, transformations, output_dir="test_images"):
    """
    テスト用の画像セットを作成する
    
    Args:
        base_image_path: 元画像のパス
        transformations: [(theta_deg, scale, name), ...] の形式
        output_dir: 出力ディレクトリ
    """
    import os
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 元画像の読み込み
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        print(f"エラー: 画像 {base_image_path} を読み込めませんでした")
        return
    
    # 元画像を円形にクロップして保存
    base_cropped = crop_to_circle(base_image)
    base_output_path = os.path.join(output_dir, "input_image.jpg")
    cv2.imwrite(base_output_path, base_cropped)
    print(f"元画像を保存: {base_output_path}")
    
    # 各変換を適用
    for theta_deg, scale, name in transformations:
        # 相似変換の適用
        transformed = apply_similarity_transform(base_image, theta_deg, scale)
        transformed_cropped = crop_to_circle(transformed)
        
        # 保存
        output_path = os.path.join(output_dir, f"output_{name}.jpg")
        cv2.imwrite(output_path, transformed_cropped)
        
        print(f"変換画像を保存: {output_path}")
        print(f"  パラメータ: θ = {theta_deg}°, s = {scale}")
        
        # プレビュー表示
        preview = np.hstack([base_cropped, transformed_cropped])
        cv2.imshow(f'Transformation: {name}', preview)
        cv2.waitKey(1000)  # 1秒表示
    
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--create-test-set":
        # テストセットの作成
        print("テスト画像セットを作成します...")
        
        # 基本画像のパス（適宜変更してください）
        base_image = "lenna.jpg"  # または任意の画像パス
        
        # テスト用の変換パラメータ
        transformations = [
            (45, 0.5, "rot45_scale05"),
            (30, 0.7, "rot30_scale07"),
            (-20, 1.2, "rot-20_scale12"),
            (60, 0.8, "rot60_scale08"),
            (0, 0.6, "rot0_scale06"),
        ]
        
        create_test_images(base_image, transformations)
        print("\nテスト画像セットが作成されました。")
        print("使用例：")
        print("python gauss_newton_estimation.py test_images/input_image.jpg test_images/output_rot45_scale05.jpg")
        
    elif len(sys.argv) == 5:
        # 単一の変換を適用
        input_path = sys.argv[1]
        theta_deg = float(sys.argv[2])
        scale = float(sys.argv[3])
        output_path = sys.argv[4]
        
        # 画像の読み込み
        image = cv2.imread(input_path)
        if image is None:
            print(f"エラー: 画像 {input_path} を読み込めませんでした")
            sys.exit(1)
        
        print(f"入力画像: {input_path}")
        print(f"変換パラメータ: θ = {theta_deg}°, s = {scale}")
        
        # 相似変換の適用
        transformed = apply_similarity_transform(image, theta_deg, scale)
        
        # 円形クロップ
        input_cropped = crop_to_circle(image)
        output_cropped = crop_to_circle(transformed)
        
        # 結果の保存
        cv2.imwrite(output_path.replace('.jpg', '_input.jpg'), input_cropped)
        cv2.imwrite(output_path, output_cropped)
        
        print(f"変換結果を保存: {output_path}")
        
        # プレビュー表示
        preview = np.hstack([input_cropped, output_cropped])
        cv2.imshow('Original vs Transformed', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("使用方法:")
        print("  単一変換: python apply_similarity_transform.py input.jpg theta_deg scale output.jpg")
        print("  テストセット作成: python apply_similarity_transform.py --create-test-set")
        print("\n例:")
        print("  python apply_similarity_transform.py lenna.jpg 45 0.5 lenna_transformed.jpg")
        sys.exit(1)

if __name__ == "__main__":
    main()