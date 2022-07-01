import cv2
import numpy as np
import matplotlib.pyplot as plt
import math



def stretch(x)->int:
    #70-90 -> 65-95
    #115-135 -> 110-140
    if x<70:
        return round(x/70*65)
    elif x<90:
        return round((x-70)/(90-70)*(95-65)+65)
    elif x<115:
        return round((x-90)/(115-90)*(110-95)+95)
    elif x<135:
        return round((x-115)/(135-115)*(140-110)+110)
    else:
        return round((x-135)/(255-135)*(255-140)+140)
        

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)

img_stretched = np.zeros((len(gray),len(gray[0])),dtype='uint8')
print("img_stretched:",img_stretched.shape)

for i in range(len(gray)):
    for ii in range(len(gray[0])):
        img_stretched[i][ii] = stretch(gray[i][ii])


index = [i for i in range(256)]
count = [0 for i in range(256)]
for i in img_stretched:
    for ii in i:
        count[ii] += 1

print(count)

plt.plot(index,count)    
plt.show()

cv2.imwrite('contrast_stretched.jpg',img_stretched)

index = [i for i in range(256)]
count = [0 for i in range(256)]
for i in range(256):
    count[i] = stretch(i)


plt.plot(index,count)    
plt.show()
