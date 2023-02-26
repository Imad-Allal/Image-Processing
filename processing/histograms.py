import numpy as np
import matplotlib.pyplot as plt

#création de l'histogramme
def func_histogram(img):

    #sauvegarder la taille de l'image
    height, width = img.shape

    histogram = np.zeros(256,)

    for i in range(height):
        for j in range(width):
            pixel = img[i, j]
            histogram[pixel] += 1
    
    return histogram

#création de l'histogramme cumulé
def func_histogram_c(histogram):
    histogram_c = np.zeros_like(histogram)
    histogram_c[0] = histogram[0]
    for i in range(1, 256):
        histogram_c[i] = histogram_c[i-1] + histogram[i]
    
    return histogram_c

#affichage des histogrammes
def plot_histogram(histogram = 0):
    plt.plot(histogram)
    plt.title("Histogramme de l'image")
    plt.show()  

def plot_histogram_c(histogram_c = 0):
    plt.plot(histogram_c)
    plt.title("Histogramme cumulé de l'image")
    plt.show()

def plot_histogram_e(histogram_e = 0):
    plt.plot(histogram_e)
    plt.title("Histogramme égalisé de l'image")
    plt.show()

def plot_histogram_e_c(histogram_e_c = 0):
    plt.plot(histogram_e_c)
    plt.title("Histogramme égalisé cumulé de l'image")
    plt.show()

