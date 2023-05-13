import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

for j in range(2):

    if j == 0:
        img = cv.imread('images/circles.png')

    elif j == 1: 
        img = cv.imread('images/cameraman.tif')

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gradient_square = []
    gradient_diamond = []

    #Element structurant carré de taille 3
    kernel_square_3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    gradient_square.append(cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel_square_3))

    #Element structurant carré de taille 7
    kernel_square_7 = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    gradient_square.append(cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel_square_7))

    #Diamond
    kernel_diamond_1 = cv.getStructuringElement(cv.MORPH_CROSS, (1, 1))
    gradient_diamond.append(cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel_diamond_1))

    kernel_diamond_5 = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    gradient_diamond.append(cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel_diamond_5))

    plt.figure(figsize=(10, 4))
    plt.subplot(141)
    plt.imshow(gradient_square[0], cmap='gray')
    plt.title('Gradient carre: 3')
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(gradient_square[1], cmap='gray')
    plt.title('Gradient carre: 7')
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(gradient_diamond[0], cmap='gray')
    plt.title('Gradient diamond: 1')
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(gradient_diamond[1], cmap='gray')
    plt.title('Gradient diamond: 5')
    plt.axis('off')
    plt.show()


    #Gradients numeriques

    gradient_sobel_square = []
    gradient_prewitt_square = []
    gradient_roberts_square = []

    for i in range(2):
        gradient_sobel_x = cv.Sobel(gradient_square[i], cv.CV_64F, 1, 0, ksize=3)
        gradient_sobel_y = cv.Sobel(gradient_square[i], cv.CV_64F, 0, 1, ksize=3)
        gradient_sobel_square.append(cv.magnitude(gradient_sobel_x, gradient_sobel_y))

        gradient_prewitt_x = cv.Sobel(gradient_square[i], cv.CV_64F, 1, 0, ksize=3)
        gradient_prewitt_y = cv.Sobel(gradient_square[i], cv.CV_64F, 0, 1, ksize=3)
        gradient_prewitt_square.append(cv.magnitude(gradient_prewitt_x, gradient_prewitt_y))

        gradient_roberts_x = cv.filter2D(gradient_square[i], -1, np.array([[1, 0], [0, -1]], dtype=np.float32))
        gradient_roberts_y = cv.filter2D(gradient_square[i], -1, np.array([[0, 1], [-1, 0]], dtype=np.float32))
        gradient_roberts_square.append(cv.magnitude(gradient_roberts_x.astype(np.float32), gradient_roberts_y.astype(np.float32)))

        plt.figure(figsize=(10, 4))
        plt.subplot(131)
        plt.imshow(gradient_sobel_square[i], cmap='gray')
        plt.title('Gradients num. Sobel carre')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(gradient_prewitt_square[i], cmap='gray')
        plt.title('Gradients num. Prewitt carre')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(gradient_roberts_square[i], cmap='gray')
        plt.title('Gradients num. Roberts carre')
        plt.axis('off')
        plt.show()

    gradient_sobel_diamond = []
    gradient_prewitt_diamond = []
    gradient_roberts_diamond = []

    for i in range(2):
        gradient_sobel_x = cv.Sobel(gradient_diamond[i], cv.CV_64F, 1, 0, ksize=3)
        gradient_sobel_y = cv.Sobel(gradient_diamond[i], cv.CV_64F, 0, 1, ksize=3)
        gradient_sobel_diamond.append(cv.magnitude(gradient_sobel_x, gradient_sobel_y))

        gradient_prewitt_x = cv.Sobel(gradient_diamond[i], cv.CV_64F, 1, 0, ksize=3)
        gradient_prewitt_y = cv.Sobel(gradient_diamond[i], cv.CV_64F, 0, 1, ksize=3)
        gradient_prewitt_diamond.append(cv.magnitude(gradient_prewitt_x, gradient_prewitt_y))

        gradient_roberts_x = cv.filter2D(gradient_diamond[i], -1, np.array([[1, 0], [0, -1]], dtype=np.float32))
        gradient_roberts_y = cv.filter2D(gradient_diamond[i], -1, np.array([[0, 1], [-1, 0]], dtype=np.float32))
        gradient_roberts_diamond.append(cv.magnitude(gradient_roberts_x.astype(np.float32), gradient_roberts_y.astype(np.float32)))

        plt.figure(figsize=(10, 4))
        plt.subplot(131)
        plt.imshow(gradient_sobel_diamond[i], cmap='gray')
        plt.title('Grad num. Sobel diamond')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(gradient_prewitt_diamond[i], cmap='gray')
        plt.title('Grad num. Prewitt diamond')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(gradient_roberts_diamond[i], cmap='gray')
        plt.title('Grad num. Roberts diamond')
        plt.axis('off')
        plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()