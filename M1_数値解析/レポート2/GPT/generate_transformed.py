import cv2
import numpy as np

def apply_similarity_transform(image, theta_deg, scale):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, theta_deg, scale)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

if __name__ == "__main__":
    img = cv2.imread("shrimp.png", cv2.IMREAD_GRAYSCALE)
    theta_deg = 45
    scale = 0.5
    transformed = apply_similarity_transform(img, theta_deg, scale)
    cv2.imwrite("transformed.png", transformed)
    print("transformed.png を保存しました。")
