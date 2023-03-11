import numpy as np
import cv2 as cv
from bruit import gaussian_noise
import lissage as ls


MASK = np.array([
    [1, 2, 3, 2, 1],
    [2, 4, 6, 4, 2],
    [3, 6, 9, 6, 3],
    [2, 4, 6, 4, 2],
    [1, 2, 3, 2, 1]], np.float32) 

print(MASK)

img = cv.imread('cameraman.tif')

#convert image from BGR (openCV format to RGB)
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
print(image.shape)

# Generate Gaussian noise
img_gauss = gaussian_noise(image)

# Lissage de l'image
img_smooth = ls.smoothing(img_gauss)

# Lissage en utilisant le masque
img_mask = ls.smoothing_mask(img_gauss, MASK)


Hori = np.concatenate((image, img_gauss, img_smooth, img_mask), axis=1)

diff = img_mask - img_smooth
cv.imwrite('new_pictures/difference.jpeg', diff)

cv.imshow('Difference des filtres', diff)
cv.imshow('Originale, Bruitee, Lissage et Lissage masque', Hori)

cv.waitKey(0)
#cv.destroyAllWindows()
