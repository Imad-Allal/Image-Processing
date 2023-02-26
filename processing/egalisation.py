import cv2 as cv
import numpy as np
import histograms as his
import measure as ms

PATH = "../pictures/img1.jpeg"

#lecture de l'image
img = cv.imread(PATH, cv.IMREAD_GRAYSCALE)
cv.imwrite('../new_pictures/egalisation/old_version.jpg', img)

#calculer les anciens les histogrammes de l'image (avant egalisation)
histogram = his.func_histogram(img)
histogram_c = his.func_histogram_c(histogram)

#anciens histogrammes
his.plot_histogram(histogram)
his.plot_histogram_c(histogram_c)

#egalisation de l'histogramme
height, width = img.shape
N = height*width
new_img = np.uint8((255 / N) * histogram_c[img])

cv.imwrite('../new_pictures/egalisation/new_version.jpg', new_img)

#calculer les nouveaux histogrammes (apres egalisation)
histogram_e = his.func_histogram(new_img)
histogram_e_c = his.func_histogram_c(histogram_e)

#nouveaux histogrammes
his.plot_histogram_e(histogram_e)
his.plot_histogram_e_c(histogram_e_c)

ms.michelson(new_img)
ms.rms(new_img)
