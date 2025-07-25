"""
相似変換プログラム：入力画像に対して回転角度とスケールを与え、中心を保ったまま相似変換を行います。


画像に対して相似変換（回転 + スケール）を適用するスクリプト。
コマンドライン引数で入力画像、スケール、回転角度を指定し、
変形後の画像を保存します。  

実行方法:
python souzi.py input_image scale angle [output_image]
- input_image: 入力画像のパス
- scale: スケール係数（例: 0.5）
- angle: 回転角度（度単位）
- output_image: 出力画像のパス（省略時は "output.png"）
"""

import sys
import cv2
import numpy as np

def apply_similarity_transform(img, scale, angle_deg):
    """
    Perform similarity transform (rotation + scale) around image center.
    """
    theta = np.deg2rad(angle_deg)
    h, w = img.shape[:2]
    cx, cy = w/2, h/2
    # a = scale*cosθ, b = scale*sinθ
    a = scale * np.cos(theta)
    b = scale * np.sin(theta)
    # translation to keep center fixed
    tx = (1 - a) * cx + b * cy
    ty = -b * cx + (1 - a) * cy
    # 2x3 transform matrix
    M = np.array([[a, -b, tx],
                  [b,  a, ty]], dtype=np.float32)
    transformed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return transformed


def main():
    if len(sys.argv) < 4:
        print("Usage: python souzi.py input_image scale angle [output_image]")
        sys.exit(1)
    input_path = sys.argv[1]
    scale = float(sys.argv[2])
    angle = float(sys.argv[3])
    output_path = sys.argv[4] if len(sys.argv) >=5 else "output.png"

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot open {{input_path}}")
        sys.exit(1)

    result = apply_similarity_transform(img, scale, angle)
    cv2.imwrite(output_path, result)
    print(f"Transformed image saved to {{output_path}}")

if __name__ == "__main__":
    main()