import cv2
import matplotlib.pyplot as plt

img_cv   = cv2.imread('1.webp')
print("img_cv:",img_cv.shape) 

gray=cv2.cvtColor(img_cv,cv2.COLOR_BGR2GRAY)
print("gray:",gray.shape)

cv2.imwrite('gray.jpg',gray)

index = [i for i in range(256)]
count = [0 for i in range(256)]
for i in gray:
    for ii in i:
        count[ii] += 1

print(count)

plt.plot(index,count)    
plt.show()

