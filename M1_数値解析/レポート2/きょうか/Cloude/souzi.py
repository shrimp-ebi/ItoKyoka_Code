"""
相似変換プログラム
入力画像に対して指定された回転角度とスケールで相似変換を適用します。

使用方法:
# カラー画像として処理（デフォルト）
python souzi.py --input Fuji.jpg --output out-Fuji-color.png --theta 30 --scale 1.5

# グレースケールとして処理
python souzi.py --input Fuji.jpg --output out-Fuji-gray.png --theta 30 --scale 1.5 --grayscale
"""

import numpy as np
import cv2
import argparse
from pathlib import Path

def similarity_transform(img, theta_deg, scale):
    """
    画像に相似変換を適用する
    
    Parameters:
    -----------
    img : numpy.ndarray
        入力画像
    theta_deg : float
        回転角度（度数法）
    scale : float
        スケールパラメータ
    
    Returns:
    --------
    transformed : numpy.ndarray
        変換後の画像
    """
    # 角度をラジアンに変換
    theta = np.deg2rad(theta_deg)
    
    # 画像サイズ
    h, w = img.shape[:2]
    
    # 画像中心
    cx = w / 2
    cy = h / 2
    
    # 変換行列の作成
    # 相似変換行列（回転＋スケール）
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 中心を原点に移動 → 相似変換 → 元の位置に戻す
    M = np.array([
        [scale * cos_theta, -scale * sin_theta, cx - scale * (cx * cos_theta - cy * sin_theta)],
        [scale * sin_theta, scale * cos_theta, cy - scale * (cx * sin_theta + cy * cos_theta)]
    ])
    
    # 変換後の画像サイズ（元のサイズと同じ）
    transformed = cv2.warpAffine(img, M, (w, h), borderValue=0)
    
    return transformed

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='画像に相似変換を適用します')
    parser.add_argument('--input', '-i', type=str, default='input.jpg',
                        help='入力画像ファイル名')
    parser.add_argument('--output', '-o', type=str, default='output.png',
                        help='出力画像ファイル名')
    parser.add_argument('--theta', '-t', type=float, default=30.0,
                        help='回転角度（度数法）')
    parser.add_argument('--scale', '-s', type=float, default=1.5,
                        help='スケールパラメータ')
    parser.add_argument('--grayscale', '-g', action='store_true',
                        help='グレースケールで処理する場合に指定')
    
    args = parser.parse_args()
    
    # 入力画像の読み込み
    # グレースケールオプションの処理
    if args.grayscale:
        img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"エラー: 画像 '{args.input}' を読み込めません")
        return
    
    print(f"入力画像: {args.input}")
    print(f"画像サイズ: {img.shape}")
    print(f"回転角度: {args.theta}°")
    print(f"スケール: {args.scale}")
    
    # 相似変換の適用
    transformed = similarity_transform(img, args.theta, args.scale)
    
    # 結果の保存
    cv2.imwrite(args.output, transformed)
    print(f"変換結果を '{args.output}' に保存しました")
    
    # 変換前後の画像を並べて表示（オプション）
    import matplotlib
    matplotlib.use('Agg')  # GUIを使わないバックエンドを指定
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # カラー画像の場合はBGRからRGBに変換して表示
    if len(img.shape) == 3:
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(transformed, cmap='gray')
    
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].set_title(f'Transformed (θ={args.theta}°, s={args.scale})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.close()  # メモリ解放
    print("比較画像を 'comparison.png' に保存しました")

if __name__ == '__main__':
    main()