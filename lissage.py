import numpy as np
import cv2 as cv
#from bruit import gaussian_noise


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
#img_gauss = gaussian_noise(image)

gauss = np.random.normal(0, 0.5, image.size)
gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')

# Add the Gaussian noise to the image
img_gauss = cv.add(image,gauss)
cv.imwrite('new_pictures/gauss.jpeg', img_gauss)

# Lissage de l'image
kernel = np.ones((5, 5), np.float32) / 25
img_smooth = cv.filter2D(img_gauss, -1, kernel)
cv.imwrite('new_pictures/img_smooth.jpeg', img_smooth)


# Lissage en utilisant le masque
img_mask = cv.filter2D(img_gauss, -1, (MASK / 81))
cv.imwrite('new_pictures/img_mask.jpeg', img_mask)

Hori = np.concatenate((image, img_gauss, img_smooth, img_mask), axis=1)

diff = img_mask - img_smooth
cv.imwrite('new_pictures/difference.jpeg', diff)

cv.imshow('Difference des filtres', diff)
cv.imshow('Originale, Bruitee, Lissage et Lissage masque', Hori)

cv.waitKey(0)
#cv.destroyAllWindows()
