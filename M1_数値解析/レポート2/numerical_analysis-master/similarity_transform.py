"""
【概要】
画像を相似変換するプログラム。

【使用方法】
入力：
・画像
・スケール
・角度
・平行移動（x軸）
・平行移動（y軸）
出力：
・変換後の画像
実行：
python similarity_transform.py {入力画像のパス} {スケール} {角度} {平行移動x} {平行移動y}
python similarity_transform.py input/color/Lenna.bmp 5 45 100 100

【情報】
作成者：勝田尚樹
作成日：2025/07/23
"""
import cv2
import numpy as np
import sys

# 変換行列を計算
def compute_M(scale, theta_rad, tx, ty):
    a = np.cos(theta_rad)
    b = np.sin(theta_rad)
    M = scale * np.array([
        [a, -b, tx],
        [b,  a, ty],
        [0,  0,  1]
    ], dtype=np.float32)
    return M

# 画像を相似変換する(順変換、補完あり)
def apply_similarity_transform_forward(img, M):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    transformed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return transformed

# 画像を相似変換する(逆変換)
def apply_similarity_transform_reverse(img, M):
    w, h = img.shape[:2]
    x_center, y_center = w/2, h/2
    M_inv = np.linalg.inv(M)
    dst = np.zeros_like(img)
    for y_dst in range(dst.shape[0]):
        for x_dst in range(dst.shape[1]):
            x_org, y_org, _ = M_inv @ np.array([x_dst-x_center, y_dst-y_center, 1])
            x_org, y_org = int(round(x_org+x_center)), int(round(y_org+y_center))
            if 0 < x_org < w and 0 < y_org < h:
                dst[y_dst][x_dst] = img[y_org][x_org]
    return dst 

# 画像を円形に切り出す
def crop_img_into_circle(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w//2, h//2)
    radius = min(center)
    cv2.circle(mask, center, radius, 255, -1)
    img_cropped = cv2.bitwise_and(img, img, mask=mask)
    return img_cropped

def main():
    # パラメータ設定
    if len(sys.argv) != 6:
        print("Usage: python script.py input_image_path scale, theta_deg, tx, ty")
        sys.exit(1)
    input_path = sys.argv[1]
    scale = float(sys.argv[2])
    theta_deg = float(sys.argv[3])
    theta_rad = np.deg2rad(theta_deg)
    tx = float(sys.argv[4])
    ty = float(sys.argv[5])
    # 画像準備
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    # 相似変換の適用
    M = compute_M(scale, theta_rad, tx, ty)
    result = apply_similarity_transform_reverse(img, M)
    # 円形に切り出し
    img_cropped = crop_img_into_circle(img)
    result_cropped = crop_img_into_circle(result)
    # 結果を表示・保存
    cv2.imshow("input", img_cropped)
    cv2.imshow("output", result_cropped)
    cv2.waitKey(0)
    cv2.imwrite("input.jpg", img_cropped)
    cv2.imwrite("output.jpg", result_cropped)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

