import numpy as np
import cv2 as cv


MASK = np.array([
    [1, 2, 3, 2, 1],
    [2, 4, 6, 4, 2],
    [3, 6, 9, 6, 3],
    [2, 4, 6, 4, 2],
    [1, 2, 3, 2, 1]], np.float32) 


img = cv.imread('cameraman.tif')

#convert image from BGR (openCV format to RGB)
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
print(image.shape)

# Generate Gaussian noise
gauss = np.random.normal(0, 0.01, image.size)
gauss = gauss.reshape(image.shape).astype('uint8')
img_noisy = cv.add(image, gauss)

# Lissage de l'image
kernel = np.ones((5, 5), np.float32) / 25
img_smooth = cv.filter2D(img_noisy, -1, kernel)

# Lissage en utilisant le masque
img_mask = cv.filter2D(img_noisy, -1, (MASK / 81))

Hori = np.concatenate((image, img_noisy, img_smooth, img_mask), axis=1)
cv.imshow('Originale, Bruitee, Lissage et Lissage masque', Hori)

diff = img_mask - img_noisy
# print(f'Filtre de lissage avec masque: {img_mask}')
# print(f'Filtre de lissage moyenneur{img_noisy}')
# print(f'Difference {diff}')
cv.imshow('Difference', diff)
cv.waitKey(0)
#cv.destroyAllWindows()
