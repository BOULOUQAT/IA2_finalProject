import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def glcm_features(image):
    """
    Extrait les caractéristiques de la matrice de co-occurrence des niveaux de gris (GLCM).
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(image_gray, [1], [0], symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return [contrast, correlation, energy, homogeneity]

def haralick_features(image):
    """
    Extrait les caractéristiques Haralick à partir de l'image.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(image_gray, [1], [0], symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return [contrast, correlation, energy, homogeneity]

def bit_features(image):
    """
    Extrait les caractéristiques BiT de l'image.
    """
    return np.random.rand(10)  # Remplacez par une vraie méthode BiT
