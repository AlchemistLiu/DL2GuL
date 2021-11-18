# Sobel边缘检测算子
import cv2
import numpy as np

# # sobel算子的实现
def sobel_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])   # X方向
    s_suanziY = np.array([[-1,-2,-1],
                          [0,0,0],
                          [1,2,1]])   
    for i in range(r-2):
        for j in range(c-2):
            new_imageX[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziX))
            # new_imageY[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziY))
            # new_image[i+1, j+1] = (new_imageX[i+1, j+1]*new_imageX[i+1,j+1] + new_imageY[i+1, j+1]*new_imageY[i+1,j+1])**0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_imageX) # 无方向算子处理的图像



img = cv2.imread('myDLstudy\DLmeet\I68.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)
# sobel 算子
out_sobel = sobel_suanzi(img)
cv2.imshow('out_sobel_image', out_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
''''''