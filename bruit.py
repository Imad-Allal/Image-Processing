import cv2 as cv
import numpy as np

img = cv.imread('cameraman.tif')

#convert image from BGR (openCV format to RGB)
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

print(image.shape)

# Generate Gaussian noise
def gaussian_noise(image):
    gauss = np.random.normal(0, 0.3, image.size)
    gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')

    # Add the Gaussian noise to the image
    img_gauss = cv.add(image,gauss)
    cv.imwrite('new_pictures/gauss.jpeg', img_gauss)

    return img_gauss

# Generate Salt & Pepper noise
def salt_pepper_noise(image):

    # Add Salt & Pepper noise to the image
    noise = np.random.randint(low=0, high=21, size = (image.shape[0], image.shape[1], 1))
    # Ajout du bruit blanc
    image_SP = np.where(noise == 40, 255, image)
    # Ajout du bruit noir
    image_SP = np.where(noise == 20, 0, image_SP)

    return image_SP

img_gauss = gaussian_noise(image)
image_SP = salt_pepper_noise(image)
cv.imwrite('new_pictures/salt_pepper.jpeg', image_SP)

# Concatener les images horizentallement
Hori = np.concatenate((image, img_gauss, image_SP), axis=1)
cv.imshow('Originale, Bruit Gaussien et Bruit Sel et Poivre', Hori)




cv.waitKey(0)