import cv2 as cv
import numpy as np
import histograms as his
import image_processing.processing.measure as ms

PATH = "../pictures/img1.jpeg"

#lecture de l'image
img = cv.imread(PATH, cv.IMREAD_GRAYSCALE)
cv.imwrite('../new_pictures/etirement/old_version.jpg', img)

#calculer les histogrammes de l'image
histogram = his.func_histogram(img)
histogram_c = his.func_histogram_c(histogram)

#anciens histogrammes
his.plot_histogram(histogram)
his.plot_histogram_c(histogram_c)

#le minimum et le maximum de l'histogramme
min_val = np.min(img)
max_val = np.max(img)

#étirement de l'histogramme
new_img = np.uint8((img - min_val) / (max_val - min_val) * 255)

#sauvegarder la nouvelle image
cv.imwrite('../new_pictures/etirement/new_version.jpg', new_img)

new_histogram = his.func_histogram(new_img)
new_histogram_c = his.func_histogram_c(new_histogram)

#nouveaux histogrammes
his.plot_histogram(new_histogram)
his.plot_histogram_c(new_histogram_c)

ms.michelson(new_img)
ms.rms(new_img)
