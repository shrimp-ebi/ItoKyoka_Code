import cv2
import numpy as np

def apply_similarity_transform(image, theta, scale):
    # 画像のサイズを取得
    (h, w) = image.shape[:2]

    # 画像の中心を計算
    center = (w / 2, h / 2)

    # 変換行列を計算 (OpenCVの関数を使用)
    M = cv2.getRotationMatrix2D(center, theta, scale)

    # 画像を変換
    transformed_image = cv2.warpAffine(image, M, (w, h))

    return transformed_image

# 入力画像の読み込み
input_image = cv2.imread('wows.jpg')

# 変換パラメータ
theta = 30  # 回転角度 (度単位)
scale = 0.7  # スケールパラメータ

# 画像に相似変換を適用
output_image = apply_similarity_transform(input_image, theta, scale)

# 結果の表示
cv2.imshow('Transformed Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 結果の保存
cv2.imwrite('output.jpg', output_image)
