"""
相似変換プログラム（Yu-gauss.py互換版）
Yu-gauss.pyと同じ座標系で相似変換を適用します。

使用方法:
python souzi2.py --input Fuji.jpg --output out-Fuji-yu.png --theta 30 --scale 1.5
"""

import numpy as np
import cv2
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def similarity_transform_yu(img, theta_deg, scale):
    """
    Yu-gauss.pyと同じ座標系で相似変換を適用
    """
    # 角度をラジアンに変換
    theta = np.deg2rad(theta_deg)
    
    # 画像サイズ
    h, w = img.shape[:2]
    
    # 画像中心
    cx = w / 2
    cy = h / 2
    
    # 出力画像
    output = np.zeros_like(img)
    
    # 各ピクセルに対して逆変換を適用
    for y in range(h):
        for x in range(w):
            # 中心を原点とした座標（Yu-gauss.pyと同じ）
            x_c = x - cx
            y_c = cy - y  # Y座標を反転
            
            # 逆変換（入力画像のどの位置から取得するか）
            x_src = (x_c * np.cos(-theta) - y_c * np.sin(-theta)) / scale
            y_src = (x_c * np.sin(-theta) + y_c * np.cos(-theta)) / scale
            
            # 画像座標系に戻す
            x_src = x_src + cx
            y_src = cy - y_src  # Y座標を再度反転
            
            # 範囲チェックと補間
            if 0 <= x_src < w-1 and 0 <= y_src < h-1:
                # バイリニア補間
                x_int = int(x_src)
                y_int = int(y_src)
                dx = x_src - x_int
                dy = y_src - y_int
                
                if len(img.shape) == 2:  # グレースケール
                    output[y, x] = (1-dx)*(1-dy)*img[y_int, x_int] + \
                                   dx*(1-dy)*img[y_int, x_int+1] + \
                                   (1-dx)*dy*img[y_int+1, x_int] + \
                                   dx*dy*img[y_int+1, x_int+1]
                else:  # カラー
                    for c in range(3):
                        output[y, x, c] = (1-dx)*(1-dy)*img[y_int, x_int, c] + \
                                          dx*(1-dy)*img[y_int, x_int+1, c] + \
                                          (1-dx)*dy*img[y_int+1, x_int, c] + \
                                          dx*dy*img[y_int+1, x_int+1, c]
    
    return output.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Yu-gauss.py互換の相似変換')
    parser.add_argument('--input', '-i', type=str, default='Fuji.jpg',
                        help='入力画像ファイル名')
    parser.add_argument('--output', '-o', type=str, default='out-Fuji-yu.png',
                        help='出力画像ファイル名')
    parser.add_argument('--theta', '-t', type=float, default=30.0,
                        help='回転角度（度数法）')
    parser.add_argument('--scale', '-s', type=float, default=1.5,
                        help='スケールパラメータ')
    parser.add_argument('--grayscale', '-g', action='store_true',
                        help='グレースケールで処理')
    
    args = parser.parse_args()
    
    # 画像読み込み
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
    print("Yu-gauss.py互換の座標系で変換を実行...")
    
    # 相似変換の適用
    transformed = similarity_transform_yu(img, args.theta, args.scale)
    
    # 保存
    cv2.imwrite(args.output, transformed)
    print(f"変換結果を '{args.output}' に保存しました")
    
    # 比較画像
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
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
    plt.savefig('comparison_yu.png')
    plt.close()
    print("比較画像を 'comparison_yu.png' に保存しました")

if __name__ == '__main__':
    main()