import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 変換パラメータ（自分で決める）
theta_deg = -30        # 回転角（度）
scale = 0.8           # 縮小率（例：0.8なら80%に縮小）
# -----------------------------

# 入力画像をグレースケールで読み込み
img = cv2.imread(".jpg", cv2.IMREAD_GRAYSCALE)
# 円形マスクを作成
h, w = img.shape
center = (w // 2, h // 2)
radius = min(center[0], center[1])
mask = np.zeros_like(img, dtype=np.uint8)
cv2.circle(mask, center, radius, 255, thickness=-1)
img_circle = cv2.bitwise_and(img, mask)
cv2.imwrite("circle_wallpaper.jpg",img_circle)
# アフィン変換行列の作成（回転＋スケーリング）
#center = (0, 0)
M = cv2.getRotationMatrix2D(center, theta_deg, scale)

# アフィン変換を適用
transformed = cv2.warpAffine(img_circle, M, (w, h), flags=cv2.INTER_LINEAR)

# 保存（必要なら）
cv2.imwrite(".jpg", transformed)
