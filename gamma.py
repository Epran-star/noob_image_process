import cv2
import numpy as np
import matplotlib.pyplot as plt
import math



def gamma(x)->int:
    return 255*math.pow(x/255,5)
    
        

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)

img_gammaed = np.zeros((len(gray),len(gray[0])),dtype='uint8')
print("img_gammaed:",img_gammaed.shape)

for i in range(len(gray)):
    for ii in range(len(gray[0])):
        img_gammaed[i][ii] = gamma(gray[i][ii])


index = [i for i in range(256)]
count = [0 for i in range(256)]
for i in img_gammaed:
    for ii in i:
        count[ii] += 1

print(count)

plt.plot(index,count)    
plt.show()

cv2.imwrite('gamma.jpg',img_gammaed)

index = [i for i in range(256)]
count = [0 for i in range(256)]
for i in range(256):
    count[i] = gamma(i)


plt.plot(index,count)    
plt.show()
