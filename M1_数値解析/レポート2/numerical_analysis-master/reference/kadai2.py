import cv2
import numpy as np

# 相似変換による入力画像の変換
def apply_similarity_transform(image, theta, scale):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, theta, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
def smooth(img):
    #水平方向微分
    kernel_1_x = np.array([[0, 0, 0],
                        [-1, 0, 1],
                        [0, 0, 0]])
    x_1 = cv2.filter2D(img, -1, kernel_1_x)
    
    #垂直方向微分
    kernel_1_y = np.array([[0, -1, 0],
                        [0, 0, 0],
                        [0, 1, 0]])
    y_1 = cv2.filter2D(img, -1, kernel_1_y)
    return x_1,y_1

# ガウス・ニュートン法による回転角度とスケールパラメータの推定
def estimate(input, output):
    theta = 0.0 #θの初期値
    scale = 1.0 #sの初期値
    max_loop = 100
    threshold = 1e-6

    all_s = []
    all_theta = []
    all_error = []

    for iteration in range(max_loop):
        # 平滑微分画像Ix'とIy'を作成
        Ix, Iy = smooth(output)
   
        # 1階微分と2階微分を計算
        IxIx = Ix * Ix
        IyIy = Iy * Iy
        IxIy = Ix * Iy
        IxIz = Ix * output
        IyIz = Iy * output

        # JJのθに対する1階微分JJθθと2階微分JJθθθθを計算
        JJ_theta_theta = np.sum(IxIx * (scale * np.sin(theta))**2 + IyIy * (scale * np.cos(theta))**2 - 2 * IxIy * scale**2 * np.sin(theta) * np.cos(theta))

        # JJのssに対する1階微分JJssと2階微分JJssssを計算
        JJ_scale_scale = np.sum(IxIx * np.sin(theta)**2 + IyIy * np.cos(theta)**2 - 2 * IxIy * np.sin(theta) * np.cos(theta))

        # JJのθとssでの微分JJθθを計算
        J_theta = np.sum(IxIz * (scale * np.sin(theta)) + IyIz * (scale * np.cos(theta)))
        J_scale = np.sum(IxIz * np.sin(theta) + IyIz * np.cos(theta))

        # ΔθとΔsを計算
        delta_theta = (JJ_scale_scale * J_theta - J_scale * JJ_theta_theta) / (JJ_theta_theta * JJ_scale_scale - J_theta**2)
        delta_scale = (J_theta - J_scale * delta_theta) / JJ_theta_theta

        # ガウス・ニュートン法による更新
        theta += delta_theta
        scale += delta_scale

        all_s.append(scale)
        all_theta.append(np.rad2deg(theta))
        # all_error.append(error)
        # 収束判定
        if np.abs(delta_theta) < threshold and np.abs(delta_scale) < threshold:
            break
    print("Estimated Theta (degrees):", np.degrees(theta))
    print("Estimated Scale:", scale)
    return theta, scale


def main(input_image_path, output_image_path):
     # テスト用の入力画像と出力画像をグレースケールで読み込みます
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow(input_image_path,input_image)
    cv2.imshow(output_image_path,output_image)
    cv2.waitKey()
    # 推定を実行
    estimated_theta, estimated_scale = estimate(input_image, output_image)

    print("Estimated Theta (degrees):", np.degrees(estimated_theta))
    print("Estimated Scale:", estimated_scale)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python kadai2.py input_image_path output_image_path")
    else:
        main(sys.argv[1], sys.argv[2])
