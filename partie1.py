import cv2 as cv
import numpy as np
import bruit as br

img = cv.imread('cameraman.tif')

#convert image from BGR (openCV format to RGB)
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

print(image.shape)

# Generate Gaussian noise
img_gauss = br.gaussian_noise(image)

# Generate Salt & Pepper noise
image_SP = br.salt_pepper_noise(image)

# Concatener les images horizentallement
Hori = np.concatenate((image, img_gauss, image_SP), axis=1)
cv.imshow('Originale, Bruit Gaussien et Bruit Sel et Poivre', Hori)

cv.waitKey(0)