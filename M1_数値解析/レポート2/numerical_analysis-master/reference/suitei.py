import numpy as np
import cv2
import csv
import math
from matplotlib import pyplot as plt
from matplotlib import cm

#試行回数
loop = 100
#初期パラメータ
theta = np.deg2rad(0)
s = 1
all_s = []
all_theta = []
all_error = []
sigma = 1.5
coef = 1/(2 * np.pi*(sigma**2))
gaussian_x = [[] for _ in range(int(1+4*sigma))]
gaussian_y = [[] for _ in range(int(1+4*sigma))]

for y in range(int(1+4*sigma)):
    offset_y = round(int(4*sigma)/2) - y
    for x in range(int(1+4*sigma)):
        offset_x = x - round(int(4*sigma)/2)
        gaussian_x[y].append(-coef * (-offset_x / sigma**2) * np.exp(-1*(offset_x**2 + offset_y**2) / (2*sigma**2) ))
        gaussian_y[y].append(-coef * (-offset_y / sigma**2) * np.exp(-1*(offset_x**2 + offset_y**2) / (2*sigma**2) ))
        
gaussian_x = np.array(gaussian_x, dtype = float)
gaussian_y = np.array(gaussian_y, dtype = float)

#画像の読み込み
img_in  = cv2.imread('wows.jpg', cv2.IMREAD_GRAYSCALE)
img_out = cv2.imread('out_wows.jpg', cv2.IMREAD_GRAYSCALE)
#画像の大きさ
pixel = img_in.shape[0]

cx_in = (img_in.shape[0])/2
cy_in = cx_in
cx_out = (img_out.shape[0])/2
cy_out = cx_in

X = []
Y = []
radius = img_in.shape[0]/2
for i in range(img_in.shape[0]):
    for j in range(img_in.shape[1]):
        length = np.sqrt((i-cx_in)**2 + (j-cy_in)**2)
        if length <= radius:
            X.append(i)
            Y.append(j)

im = np.zeros(img_in.shape, dtype = float)
Ix = np.zeros(img_in.shape, dtype = float)
Iy = np.zeros(img_in.shape, dtype = float)

for i in range(loop):
    J_theta = 0
    J_2theta = 0
    J_s = 0
    J_2s = 0
    J_theta_s = 0
    error = 0

    #現パラメータのθ分だけ回転させる
    for j in range(len(X)):
        x = X[j]
        y = Y[j]
        x_rot = round( ((x-cx_in) * np.cos(-theta) - (y-cy_in) * np.sin(-theta) ) + cx_in)
        y_rot = round( ((x-cx_in) * np.sin(-theta) + (y-cy_in) * np.cos(-theta) ) + cy_in)
        if x_rot >= pixel or y_rot >= pixel:
            im[y, x] = 0
        else:
            #print(img_out.shape)
            im[y, x] = img_out[y_rot, x_rot]
            
    #平滑微分計算
    im = im.astype(np.float64)
    Ixt = cv2.filter2D(im, -1, gaussian_x)
    Iyt = cv2.filter2D(im, -1, gaussian_y)

    #画像を元の座標系に戻す
    for j in range(len(X)):
        x = X[j]
        y = Y[j]
        x_rot = round( ((x-cx_in) * np.cos(theta) - (y-cy_in) * np.sin(theta)) + cx_in)
        y_rot = round( ((x-cx_in) * np.sin(theta) + (y-cy_in) * np.cos(theta)) + cy_in)
        if x_rot >= pixel or y_rot >= pixel:
            Ix[y, x] = 0
            Iy[y, x] = 0
        else:
            Ix[y, x] = Ixt[y_rot, x_rot]
            Iy[y, x] = Iyt[y_rot, x_rot]

    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot(121)
    ax1.imshow(Ix)
    ax2 = fig.add_subplot(122)
    ax1.imshow(Iy)
    plt.savefig("{}.png".format(i))
    plt.close()

    for j in range(len(X)):
        x = X[j]
        y = Y[j]

        dif_x = round( s *  ( (x - cx_in) * np.cos(theta) - (cy_in-y) * np.sin(theta) ) +cx_in)
        dif_y = round( -s * ( (x - cx_in) * np.sin(theta) + (cy_in-y) * np.cos(theta) ) +cy_in)

        if dif_x >= pixel or dif_y >= pixel or dif_x < 0 or dif_y < 0:
            dif_I = 0
            im[y,x] = 0
            dif_Ix = 0
            dif_Iy = 0
        else:
            dif_I = img_out[dif_y, dif_x]
            im[y,x] = img_out[dif_y, dif_x]
            dif_Ix = Ix[dif_y, dif_x]
            dif_Iy = Iy[dif_y, dif_x]
            #オーバーフロー防止
            dif_I = dif_I.astype('int64')
        I = img_in[y,x]

        #x,yをthrtaで微分
        dx_theta = (s* (-(x-cx_in) * np.sin(theta) - (cy_in-y) * np.cos(theta)))
        dy_theta = (s* ( (x-cx_in) * np.cos(theta) - (cy_in-y) * np.sin(theta)))
        #x,yをsで微分
        dx_s = ((x-cx_in) * np.cos(theta) - (cy_in-y) * np.sin(theta))
        dy_s = ((x-cx_in) * np.sin(theta) + (cy_in-y) * np.cos(theta))
        #x,yをthetaとsで微分
        dx_theta_s = (-(x-cx_in) * np.sin(theta) - (cy_in-y) * np.cos(theta))
        dy_theta_s = ( (x-cx_in) * np.cos(theta) - (cy_in-y) * np.sin(theta))
        #オーバーフロー防止
        I = I.astype('int64')
        
        #thetaでの微分式
        J_theta = J_theta + ( (dif_I - I) * (dif_Ix * dx_theta + dif_Iy * dy_theta) )
        J_2theta = J_2theta + (dif_Ix * dx_theta + dif_Iy * dy_theta)**2
        #sでの微分式
        J_s_temp = (dif_I - I) * (dif_Ix * dx_s + dif_Iy * dy_s)
        J_s = J_s + J_s_temp
        J_2s_temp = (dif_Ix * dx_s + dif_Iy * dy_s)**2
        J_2s = J_2s + J_2s_temp                  
        #thetaとsで微分
        J_theta_s = J_theta_s + dif_Ix**2 * dx_theta * dx_s + dif_Ix * dif_Ix * ( dx_theta * dy_s + dx_s * dy_theta) + dif_Iy**2 * dy_theta * dy_s

        error = error + (1/2) * (dif_I - I)**2

    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_in)
    ax2 = fig.add_subplot(122)
    ax1.imshow(im)
    plt.savefig("{}.png".format(i))
    plt.close()

    J = np.array([
                [J_2theta, J_theta_s],
                [J_theta_s, J_2s]])
    J_vec = np.array([[J_theta],[J_s]])
    print(J)
    #print(J_vec)

    [dtheta, ds] = np.matmul(np.linalg.inv(J), J_vec)

    theta = theta - dtheta[0]
    s = s -ds[0]
    print(s)
    print(theta)
    all_s.append(s)
    all_theta.append(np.rad2deg(theta))
    all_error.append(error)

print(np.rad2deg(theta))
print(s)
    
np.savetxt("s_-15065.csv", all_s, delimiter =",",fmt ='% s')
np.savetxt("theta_-15065.csv", all_theta, delimiter =",",fmt ='% s')