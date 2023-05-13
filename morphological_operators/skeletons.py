import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis

img = cv.imread('images/circles.png', 0)

#Rendre l'image en binaire
ret, binary_image = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
binary_image = binary_image.astype(np.uint8) // 255

#Inversion de l'image binaire pour la squelettisation
binary_image = 1 - binary_image

#Squelette d4
skeleton_d4 = skeletonize(binary_image)

#Squelette d8
skeleton_d8 = medial_axis(binary_image)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Originale')
plt.axis('off')
plt.subplot(132)
plt.imshow(skeleton_d4, cmap='gray')
plt.title('Squelette d4')
plt.axis('off')
plt.subplot(133)
plt.imshow(skeleton_d8, cmap='gray')
plt.title('Squelette d8')
plt.axis('off')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()