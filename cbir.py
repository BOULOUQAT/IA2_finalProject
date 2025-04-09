import os
import numpy as np
import cv2
from descriptors import glcm_features, haralick_features
from scipy.spatial.distance import euclidean, cityblock, chebyshev, canberra
import streamlit as st

dataset_path = "datasets/"  # Dossier contenant les images pour la recherche

def load_images():
    """
    Charge les images depuis le dossier `datasets/`.
    """
    images = []
    filenames = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            images.append(img)
            filenames.append(filename)
    return images, filenames

def calculate_similarity(feature1, feature2, distance_metric='euclidean'):
    """
    Calcule la similarité entre deux caractéristiques d'images en utilisant une mesure de distance.
    """
    if distance_metric == 'euclidean':
        return euclidean(feature1, feature2)
    elif distance_metric == 'manhattan':
        return cityblock(feature1, feature2)
    elif distance_metric == 'chebyshev':
        return chebyshev(feature1, feature2)
    elif distance_metric == 'canberra':
        return canberra(feature1, feature2)

def show_similar_images(query_img, num_results=5, distance_metric="euclidean", descriptor="glcm"):
    """
    Affiche les images similaires basées sur le descripteur choisi et la mesure de distance.
    """
    # Charger toutes les images du dataset
    images, filenames = load_images()
    
    # Extraire les caractéristiques de l'image de requête
    query_features = glcm_features(query_img) if descriptor == "glcm" else haralick_features(query_img)

    # Calculer les distances entre l'image de requête et les images du dataset
    distances = []
    for img in images:
        features = glcm_features(img) if descriptor == "glcm" else haralick_features(img)
        distance = calculate_similarity(query_features, features, distance_metric)
        distances.append(distance)

    # Trier les distances et afficher les images similaires
    sorted_indices = np.argsort(distances)
    for i in range(num_results):
        st.image(images[sorted_indices[i]], caption=f"Image similaire {i + 1}: {filenames[sorted_indices[i]]}")
