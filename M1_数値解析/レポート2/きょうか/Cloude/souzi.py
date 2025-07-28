"""
相似変換プログラム
入力画像に対して指定された回転角度とスケールで相似変換を行う

使用方法:
python souzi.py -i Fuji.jpg -o outFuji.png -r 30 -s 1.5
"""

import numpy as np
import cv2
import argparse

def similarity_transform(img, theta_deg, scale):
    """
    画像に相似変換を適用する
    
    Parameters:
    img: 入力画像
    theta_deg: 回転角度（度）
    scale: スケール係数
    
    Returns:
    transformed_img: 変換後の画像
    """
    height, width = img.shape[:2]
    
    # 画像の中心
    cx = width / 2.0
    cy = height / 2.0
    
    # 度からラジアンに変換
    theta = np.deg2rad(theta_deg)
    
    # 相似変換行列の作成
    # M = [[s*cos(θ), -s*sin(θ), tx],
    #      [s*sin(θ),  s*cos(θ), ty]]
    # ここでtx, tyは中心を原点として変換してから元に戻すための並進
    
    # 回転・スケール行列
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 変換行列（中心を原点に移動→回転・スケール→元の位置に戻す）
    M = np.array([
        [scale * cos_theta, -scale * sin_theta, cx - scale * (cx * cos_theta - cy * sin_theta)],
        [scale * sin_theta,  scale * cos_theta, cy - scale * (cx * sin_theta + cy * cos_theta)]
    ])
    
    # 出力画像のサイズを計算（スケールを考慮）
    output_size = (int(width * scale * 1.5), int(height * scale * 1.5))
    
    # 出力画像の中心に配置するための調整
    tx = (output_size[0] - width) / 2.0
    ty = (output_size[1] - height) / 2.0
    M[0, 2] += tx
    M[1, 2] += ty
    
    # アフィン変換を適用
    transformed_img = cv2.warpAffine(img, M, output_size, 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
    
    return transformed_img

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='画像に相似変換を適用')
    parser.add_argument('-i', '--input', required=True, help='入力画像ファイル名')
    parser.add_argument('-o', '--output', required=True, help='出力画像ファイル名')
    parser.add_argument('-r', '--rotation', type=float, default=0, 
                       help='回転角度（度）（デフォルト: 0）')
    parser.add_argument('-s', '--scale', type=float, default=1.0, 
                       help='スケール係数（デフォルト: 1.0）')
    
    args = parser.parse_args()
    
    # 画像の読み込み
    img = cv2.imread(args.input)
    if img is None:
        print(f"エラー: 画像 '{args.input}' を読み込めません")
        return
    
    # グレースケール画像の場合も対応
    if len(img.shape) == 2:
        img_gray = img
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"入力画像: {args.input}")
    print(f"回転角度: {args.rotation}度")
    print(f"スケール: {args.scale}")
    
    # 相似変換を適用
    transformed_img = similarity_transform(img_gray, args.rotation, args.scale)
    
    # 結果を保存
    cv2.imwrite(args.output, transformed_img)
    print(f"出力画像を保存しました: {args.output}")
    
    # 表示部分を削除（SSH環境でのエラー回避）
    # 必要に応じて以下のコメントを外して使用
    # cv2.imshow('Original', img_gray)
    # cv2.imshow('Transformed', transformed_img)
    # print("任意のキーを押すと終了します...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()