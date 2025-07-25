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
import numpy as np
from PIL import Image

def apply_similarity_transform(img_arr, scale, angle_deg):
    theta = np.deg2rad(angle_deg)
    H, W = img_arr.shape[:2]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    a = scale * np.cos(theta)
    b = scale * np.sin(theta)
    ys, xs = np.indices((H, W))
    xs_c = xs - cx
    ys_c = ys - cy
    src_x =  a * xs_c + b * ys_c + cx
    src_y = -b * xs_c + a * ys_c + cy
    src_xi = np.clip(np.round(src_x).astype(int), 0, W - 1)
    src_yi = np.clip(np.round(src_y).astype(int), 0, H - 1)
    if img_arr.ndim == 2:
        transformed = img_arr[src_yi, src_xi]
    else:
        transformed = img_arr[src_yi, src_xi, :]
    return transformed


def main():
    if len(sys.argv) < 4:
        print("Usage: python souzi.py input_image scale angle_deg [output_image]")
        sys.exit(1)
    input_path, scale, angle = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    output_path = sys.argv[4] if len(sys.argv) >= 5 else "output.png"
    img = Image.open(input_path)
    img_arr = np.array(img)
    result_arr = apply_similarity_transform(img_arr, scale, angle)
    Image.fromarray(result_arr).save(output_path)
    print(f"Transformed image saved to {output_path}")

if __name__ == "__main__":
    main()
