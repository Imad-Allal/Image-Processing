import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("text.png", cv.IMREAD_GRAYSCALE)

#image = plt.imread("text.png")

# Élément structurant en forme de ligne
kernel_line_3 = np.array([[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]], dtype=np.uint8)

kernel_line_5 = np.array([[0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0]], dtype=np.uint8)

kernel_line_7 = np.array([[0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)

# Élément structurant en forme de carré
kernel_square_3 = np.ones((3, 3), dtype=np.uint8)

kernel_square_5 = np.ones((5, 5), dtype=np.uint8)

kernel_square_7 = np.ones((7, 7), dtype=np.uint8)

# Élément structurant en forme de cercle
kernel_circle_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

kernel_circle_5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

kernel_circle_7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))


# Dilatation avec l'élément structurant en forme de ligne
dilation_line_3 = cv.dilate(img, kernel_line_3, iterations=1)
dilation_line_5 = cv.dilate(img, kernel_line_5, iterations=1)
dilation_line_7 = cv.dilate(img, kernel_line_7, iterations=1)

cv.imshow('Dilation with line kernel 3x3', dilation_line_3)
cv.imshow('Dilation with line kernel 5x5', dilation_line_5)
cv.imshow('Dilation with line kernel 7x7', dilation_line_7)

# Dilatation avec l'élément structurant en forme de carré
dilation_square_3 = cv.dilate(img, kernel_square_3, iterations=1)
dilation_square_5 = cv.dilate(img, kernel_square_5, iterations=1)
dilation_square_7 = cv.dilate(img, kernel_square_7, iterations=1)

cv.imshow('Dilation with square kernel 3x3', dilation_square_3)
cv.imshow('Dilation with square kernel 5x5', dilation_square_5)
cv.imshow('Dilation with square kernel 7x7', dilation_square_7)

# Dilatation avec l'élément structurant en forme de cercle
dilation_circle_3 = cv.dilate(img, kernel_circle_3, iterations=1)
dilation_circle_5 = cv.dilate(img, kernel_circle_5, iterations=1)
dilation_circle_7 = cv.dilate(img, kernel_circle_7, iterations=1)

cv.imshow('Dilation with circle kernel 3x3', dilation_circle_3)
cv.imshow('Dilation with circle kernel 5x5', dilation_circle_5)
cv.imshow('Dilation with circle kernel 7x7', dilation_circle_7)

cv.waitKey(0)
cv.destroyAllWindows()