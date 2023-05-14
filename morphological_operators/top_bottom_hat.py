import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis

I = cv.imread('images/flou_test.png', cv.IMREAD_GRAYSCALE)

#Rendre l'image en binaire
_, I2 = cv.threshold(I, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

#Appliquer le top-hat sur l'image
kernel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
I3 = cv.morphologyEx(I, cv.MORPH_TOPHAT, kernel)

#Render l'image top-hat en binaire
_, I4 = cv.threshold(I3, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

plt.figure(figsize=(10, 4))
plt.subplot(141)
plt.imshow(I, cmap='gray')
plt.title('Originale I')
plt.axis('off')
plt.subplot(142)
plt.imshow(I2, cmap='gray')
plt.title('Binaire I2')
plt.axis('off')
plt.subplot(143)
plt.imshow(I3, cmap='gray')
plt.title('Top-hat I3')
plt.axis('off')
plt.subplot(144)
plt.imshow(I4, cmap='gray')
plt.title('Binaire de Top-hat I4')
plt.axis('off')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
