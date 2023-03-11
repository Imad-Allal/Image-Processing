import numpy as np
import cv2 as cv

# Lissage de l'image
def smoothing(image, t = 5):
    kernel = np.ones((t, t), np.float32) / (t*t)
    img_smooth = cv.filter2D(image, -1, kernel)
    cv.imwrite('new_pictures/img_smooth.jpeg', img_smooth)
    return img_smooth


# Lissage en utilisant le masque
def smoothing_mask(image, MASK):
    img_mask = cv.filter2D(image, -1, (MASK / 81))
    cv.imwrite('new_pictures/img_mask.jpeg', img_mask)
    return img_mask

