import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
orb = cv2.ORB_create(nfeatures=3000)
# 画像の読み込み（グレースケール）
image_name = ""#画像の名前
img1 = cv2.imread(f'circle_{image_name}.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f'transformed_{image_name}.jpg', cv2.IMREAD_GRAYSCALE)
mean = 0
stddev = 0.5  # 標準偏差（ノイズの強さ）
noise = np.random.normal(mean, stddev, img2.shape)
noisy_img = np.clip(img2 + noise, 0, 1)
def residual(p, src, dst):
    theta, s = p
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    transformed = s * (src @ R.T)
    return (transformed - dst).ravel()

def gauss_newton_similarity(src_pts, dst_pts, image_shape, max_iter=500, tol=1e-15):
    # 画像の中心座標を取得
    h, w = image_shape
    cx, cy = w / 2, h / 2
    thetas = []
    ss = []
    delta_ps=[]
    # 中心を原点とする座標系に変換
    src = src_pts - np.array([cx, cy])
    dst = dst_pts - np.array([cx, cy])

    # 初期値 θ=0, s=1（または推定に近い初期値）
    
    theta, s = 0.0, 0.0
    deg = 0
    theta = np.radians(deg)
    for i in range(max_iter):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]])

        pred = s * (src @ R.T)
        r = (pred - dst).reshape(-1)
        J = []
        thetas.append(np.degrees(theta))
        ss.append(s)
        for x, y in src:
            dx_dtheta = s * (-x * sin_t - y * cos_t)
            dy_dtheta = s * (x * cos_t - y * sin_t)
            dx_ds     = cos_t * x - sin_t * y
            dy_ds     = sin_t * x + cos_t * y
            J.append([dx_dtheta, dx_ds])
            J.append([dy_dtheta, dy_ds])
        J = np.array(J)
        delta_p = np.linalg.lstsq(J, -r, rcond=None)[0]
        theta += delta_p[0]
        s     += delta_p[1]
        delta_ps.append(delta_p)
        print(len(delta_ps),len(thetas))
        print(f"Iteration {i+1}: θ={np.degrees(theta):.2f}°, s={s:.4f}")
        if np.linalg.norm(delta_p) < tol:
            break
    delta_ps.append(delta_p)
    thetas.append(np.degrees(theta))
    ss.append(s)
    #print(len(delta_ps),len(thetas))
    #print(delta_ps,thetas,ss,residual)
    return thetas, ss, delta_ps

h, w = img1.shape 
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# マッチング
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 対応点ペアを作成
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches if m.distance < 20])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches if m.distance < 20])

# 結果の可視化
img_out = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for p1, p2 in zip(pts1, pts2):
    cv2.circle(img1, tuple(p1.astype(int)), 3, 255, -1)
    cv2.circle(img_out, tuple(p2.astype(int)), 3, (0, 255, 0), -1)
thetas, ss, delta_ps = gauss_newton_similarity(pts1, pts2,image_shape=(h,w))
fig = plt.figure()
print(np.array(thetas))
print(np.array(ss))

# print(np.array(hessians))
delta_theta = [dp[0] for dp in delta_ps[:-1]]
delta_s     = [dp[1] for dp in delta_ps]

# 表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(img_out)
plt.tight_layout()
plt.show()
