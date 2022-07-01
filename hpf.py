import cv2
import numpy as np
import matplotlib.pyplot as plt

def hpf(dft_shift, r=100):
    m, n, _ = dft_shift.shape
    center = (m//2, n//2)
    mask = np.zeros_like(dft_shift)
    x_arr = np.concatenate([np.arange(m).reshape(m, 1)], axis=1)
    y_arr = np.concatenate([np.arange(n).reshape(1, n)], axis=0)
    dist = np.sqrt((x_arr - center[0])**2 + (y_arr - center[1])**2)
    mask[dist >= r] = 1
    return mask

def bw_hpf(dft_shift, r=100, N=1):
    m, n, _ = dft_shift.shape
    center = (m//2, n//2)
    mask = np.ones_like(dft_shift)
    x_arr = np.concatenate([np.arange(m).reshape(m, 1)], axis=1)
    y_arr = np.concatenate([np.arange(n).reshape(1, n)], axis=0)
    dist = np.sqrt((x_arr - center[0])**2 + (y_arr - center[1])**2).reshape(m, n, 1)
    mask = 1-1/(1+(dist/r)**(2*N))
    return mask

def gaussian_hpf(dft_shift, r=100):
    m, n, _ = dft_shift.shape
    center = (m//2, n//2)
    x_arr = np.concatenate([np.arange(m).reshape(m, 1)], axis=1)
    y_arr = np.concatenate([np.arange(n).reshape(1, n)], axis=0)
    dist_square = np.sqrt((x_arr - center[0])**2 + (y_arr - center[1])**2).reshape(m, n, 1)
    mask = 1-np.exp(-1*dist_square/(2*r*r))
    return mask

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)

cv2.imwrite('gray.jpg',gray)

#傅里叶变换
dft = cv2.dft(np.float32(gray), flags = cv2.DFT_COMPLEX_OUTPUT)

#将频谱低频从左上角移动至中心位置
dft_shift = np.fft.fftshift(dft)

#频谱图像双通道复数转换为0-255区间
result = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

#显示图像
plt.subplot(241)
plt.imshow(result, cmap = 'gray')
plt.xticks([]), plt.yticks([])

plt.subplot(245)
plt.axis('off')
plt.imshow(gray, cmap='gray')

#理想hpf
huv = hpf(dft_shift)                        # 转移函数（算子/模)
hpf_dft_shift = dft_shift * huv             # 低通滤波
hpf_magnitude_spectrum = cv2.magnitude(hpf_dft_shift[:,:,0], hpf_dft_shift[:,:,1])
log_hpf_magnitude_spectrum = 20*np.log(hpf_magnitude_spectrum+1) # 幅值对数变换
hpf_dft = np.fft.ifftshift(hpf_dft_shift)         # 还原频谱图
img_ = cv2.idft(hpf_dft)                           # 逆傅里叶变换
img_hpf = cv2.magnitude (img_[:,:,0],img_[:,:,1])  # 还原图像

plt.subplot(242)
plt.axis('off')
plt.imshow(log_hpf_magnitude_spectrum, cmap = 'gray')
plt.xticks([]), plt.yticks([])


plt.subplot(246)
plt.axis('off')
plt.imshow(img_hpf, cmap='gray')



#巴特沃斯低通滤波
huv = bw_hpf(dft_shift)                        # 转移函数（算子/模)
hpf_dft_shift = dft_shift * huv             # 低通滤波
hpf_magnitude_spectrum = cv2.magnitude(hpf_dft_shift[:,:,0], hpf_dft_shift[:,:,1])
log_hpf_magnitude_spectrum = 20*np.log(hpf_magnitude_spectrum+1) # 幅值对数变换
hpf_dft = np.fft.ifftshift(hpf_dft_shift)         # 还原频谱图
img_ = cv2.idft(hpf_dft)                           # 逆傅里叶变换
img_hpf = cv2.magnitude (img_[:,:,0],img_[:,:,1])  # 还原图像

plt.subplot(243)
plt.axis('off')
plt.imshow(log_hpf_magnitude_spectrum, cmap = 'gray')
plt.xticks([]), plt.yticks([])


plt.subplot(247)
plt.axis('off')
plt.imshow(img_hpf, cmap='gray')


#高斯低通滤波
huv = gaussian_hpf(dft_shift)                        # 转移函数（算子/模)
hpf_dft_shift = dft_shift * huv             # 低通滤波
hpf_magnitude_spectrum = cv2.magnitude(hpf_dft_shift[:,:,0], hpf_dft_shift[:,:,1])
log_hpf_magnitude_spectrum = 20*np.log(hpf_magnitude_spectrum+1) # 幅值对数变换
hpf_dft = np.fft.ifftshift(hpf_dft_shift)         # 还原频谱图
img_ = cv2.idft(hpf_dft)                           # 逆傅里叶变换
img_hpf = cv2.magnitude (img_[:,:,0],img_[:,:,1])  # 还原图像

plt.subplot(244)
plt.axis('off')
plt.imshow(log_hpf_magnitude_spectrum, cmap = 'gray')
plt.xticks([]), plt.yticks([])


plt.subplot(248)
plt.axis('off')
plt.imshow(img_hpf, cmap='gray')
plt.show()
