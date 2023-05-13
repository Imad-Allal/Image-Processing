import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

img = cv.imread("images/text.png", cv.IMREAD_GRAYSCALE)

#Élément structurant en forme de ligne
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

#Élément structurant en forme de carré
kernel_square_3 = np.ones((3, 3), dtype=np.uint8)
kernel_square_5 = np.ones((5, 5), dtype=np.uint8)
kernel_square_7 = np.ones((7, 7), dtype=np.uint8)

#Élément structurant en forme de cercle
kernel_circle_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel_circle_5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
kernel_circle_7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))


#Dilatation avec l'élément structurant en forme de ligne
dilation_line_3 = cv.dilate(img, kernel_line_3, iterations=1)
dilation_line_5 = cv.dilate(img, kernel_line_5, iterations=1)
dilation_line_7 = cv.dilate(img, kernel_line_7, iterations=1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(dilation_line_3, cmap='gray')
plt.title('Dilatation ligne: 3x3')
plt.axis('off')
plt.subplot(132)
plt.imshow(dilation_line_5, cmap='gray')
plt.title('Dilatation ligne: 5x5')
plt.axis('off')
plt.subplot(133)
plt.imshow(dilation_line_7, cmap='gray')
plt.title('Dilatation ligne: 7x7')
plt.axis('off')
plt.show()


#Dilatation avec l'élément structurant en forme de carré
dilation_square_3 = cv.dilate(img, kernel_square_3, iterations=1)
dilation_square_5 = cv.dilate(img, kernel_square_5, iterations=1)
dilation_square_7 = cv.dilate(img, kernel_square_7, iterations=1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(dilation_square_3, cmap='gray')
plt.title('Dilatation carre: 3x3')
plt.axis('off')
plt.subplot(132)
plt.imshow(dilation_square_5, cmap='gray')
plt.title('Dilatation carre: 5x5')
plt.axis('off')
plt.subplot(133)
plt.imshow(dilation_square_7, cmap='gray')
plt.title('Dilatation carre: 7x7')
plt.axis('off')
plt.show()

#Dilatation avec l'élément structurant en forme de cercle
dilation_circle_3 = cv.dilate(img, kernel_circle_3, iterations=1)
dilation_circle_5 = cv.dilate(img, kernel_circle_5, iterations=1)
dilation_circle_7 = cv.dilate(img, kernel_circle_7, iterations=1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(dilation_circle_3, cmap='gray')
plt.title('Dilatation cercle: 3x3')
plt.axis('off')
plt.subplot(132)
plt.imshow(dilation_circle_5, cmap='gray')
plt.title('Dilatation cercle: 5x5')
plt.axis('off')
plt.subplot(133)
plt.imshow(dilation_circle_7, cmap='gray')
plt.title('Dilatation cercle: 7x7')
plt.axis('off')
plt.show()


# Erosion avec l'élément structurant en forme de ligne
erosion_line_3 = cv.erode(img, kernel_line_3, iterations=1)
erosion_line_5 = cv.erode(img, kernel_line_5, iterations=1)
erosion_line_7 = cv.erode(img, kernel_line_7, iterations=1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(erosion_line_3, cmap='gray')
plt.title('Erosion ligne: 3x3')
plt.axis('off')
plt.subplot(132)
plt.imshow(erosion_line_5, cmap='gray')
plt.title('Erosion ligne: 5x5')
plt.axis('off')
plt.subplot(133)
plt.imshow(erosion_line_7, cmap='gray')
plt.title('Erosion ligne: 7x7')
plt.axis('off')
plt.show()

# Erosion avec l'élément structurant en forme de carré
erosion_square_3 = cv.erode(img, kernel_square_3, iterations=1)
erosion_square_5 = cv.erode(img, kernel_square_5, iterations=1)
erosion_square_7 = cv.erode(img, kernel_square_7, iterations=1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(erosion_square_3, cmap='gray')
plt.title('Erosion carre: 3x3')
plt.axis('off')
plt.subplot(132)
plt.imshow(erosion_square_5, cmap='gray')
plt.title('Erosion carre: 5x5')
plt.axis('off')
plt.subplot(133)
plt.imshow(erosion_square_7, cmap='gray')
plt.title('Erosion carre: 7x7')
plt.axis('off')
plt.show()

erosion_circle_3 = cv.erode(img, kernel_circle_3, iterations=1)
erosion_circle_5 = cv.erode(img, kernel_circle_5, iterations=1)
erosion_circle_7 = cv.erode(img, kernel_circle_7, iterations=1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(erosion_circle_3, cmap='gray')
plt.title('Erosion cercle: 3x3')
plt.axis('off')
plt.subplot(132)
plt.imshow(erosion_circle_5, cmap='gray')
plt.title('Erosion cercle: 5x5')
plt.axis('off')
plt.subplot(133)
plt.imshow(erosion_circle_7, cmap='gray')
plt.title('Erosion cercle: 7x7')
plt.axis('off')
plt.show()

#Dilatation suivie de l'érosion
dilate_erode = cv.erode(cv.dilate(img, kernel_square_3), kernel_square_3)

#Vérifier si l'image est restaurée
if np.array_equal(img, dilate_erode):
    print("La dilatation suivie de l'érosion restaure l'image d'origine.")
else:
    print("La dilatation suivie de l'érosion ne restaure pas l'image d'origine.")

#Erosion suivie de la dilatation
erode_dilate = cv.dilate(cv.erode(img, kernel_square_3), kernel_square_3)

#Vérifier si l'image est restaurée
if np.array_equal(img, erode_dilate):
    print("L'érosion suivie de la dilatation restaure l'image d'origine.")
else:
    print("L'érosion suivie de la dilatation ne restaure pas l'image d'origine.")


# Appliquer l'ouverture avec différents éléments structurants
opening_line = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_line_3)
opening_square = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_square_3)
opening_circle = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_circle_3)

# Afficher les résultats de l'ouverture
plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(opening_line, cmap='gray')
plt.title('Ouverture (ligne)')
plt.axis('off')
plt.subplot(132)
plt.imshow(opening_square, cmap='gray')
plt.title('Ouverture (carré)')
plt.axis('off')
plt.subplot(133)
plt.imshow(opening_circle, cmap='gray')
plt.title('Ouverture (cercle)')
plt.axis('off')
plt.show()

# Appliquer la fermeture avec différents éléments structurants
closing_line = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_line_3)
closing_sqaure = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_square_3)
closing_circle = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_circle_3)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(closing_line, cmap='gray')
plt.title('Fermeture (ligne)')
plt.axis('off')
plt.subplot(132)
plt.imshow(closing_sqaure, cmap='gray')
plt.title('Fermeture (carré)')
plt.axis('off')
plt.subplot(133)
plt.imshow(closing_circle, cmap='gray')
plt.title('Fermeture (cercle)')
plt.axis('off')
plt.show()

# Appliquer l'ouverture multiple
multiple_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_square_3, iterations=3)

# Appliquer la fermeture multiple
multiple_closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_square_3, iterations=3)

# Vérifier l'idempotence de l'ouverture et de la fermeture
if np.array_equal(multiple_opening, opening_square) and np.array_equal(multiple_closing, closing_sqaure):
    print("L'ouverture et la fermeture sont idempotentes.")
else:
    print("L'ouverture et la fermeture ne sont pas idempotentes.")

cv.waitKey(0)
cv.destroyAllWindows()