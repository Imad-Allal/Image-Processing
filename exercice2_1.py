import cv2 as cv
import numpy as np
import bruit as br
import lissage as ls
import matplotlib.pyplot as plt


img = cv.imread('cameraman.tif')

#convert image from BGR (openCV format to RGB)
image = cv.cvtColor(img, cv.COLOR_BGR2RGB)


#Question 1

psnr_values = []

image_SP = br.salt_pepper_noise(image)

img_smooth_3 = ls.smoothing(image_SP, 3)
psnr1 = cv.PSNR(img, img_smooth_3)
psnr_values.append(psnr1)

img_smooth_5 = ls.smoothing(image_SP, 5)
psnr2 = cv.PSNR(img, img_smooth_5)
psnr_values.append(psnr2)

img_smooth_7 = ls.smoothing(image_SP, 7)
psnr3 = cv.PSNR(img, img_smooth_7)
psnr_values.append(psnr3)

Hori = np.concatenate((image, image_SP, img_smooth_3, img_smooth_5, img_smooth_7), axis=1)

cv.imshow('Originale, Bruitee, Lissage x3, x5 et x7', Hori)

plt.plot([3, 5, 7], psnr_values)
plt.xlabel('Taille du filtre')
plt.ylabel('PSNR')
plt.title('PSNR vs. Taille du filtre (Lissage par moyenne)')
plt.show()




cv.waitKey(0)

