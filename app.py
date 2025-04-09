import streamlit as st
import cv2
import numpy as np
from PIL import Image  # Utilisation de Pillow pour gérer les fichiers d'image
import sqlite3
import dlib
from database import get_connection
import bcrypt
import os
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import euclidean, cityblock, chebyshev, canberra

# Initialiser le détecteur de visage de Dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Assurez-vous d'avoir téléchargé ce fichier

# Fonction pour capturer une seule image via la webcam
def capture_single_image():
    cap = cv2.VideoCapture(0)  # Démarre la caméra
    ret, frame = cap.read()

    if not ret:
        st.write("Impossible de capturer l'image")
        cap.release()
        return None

    # Affichage de l'image capturée dans Streamlit
    st.image(frame, channels="BGR", caption="Image capturée")

    cap.release()  # Libère la caméra
    return frame  # Retourne l'image capturée

# Fonction pour vérifier si l'utilisateur ou l'email existe déjà
def check_user_exists(username, email):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT username, email FROM users WHERE username = ? OR email = ?", (username, email))
    result = cursor.fetchone()

    conn.close()
    return result is not None

# Fonction pour extraire les descripteurs faciaux d'une image
def get_face_encoding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        st.error("Aucun visage détecté dans l'image.")
        return None

    face = faces[0]  # Utiliser le premier visage détecté
    shape = sp(gray, face)  # Prédire les points de repère du visage

    # Extraire les points de repère (coordonnées des 68 points du visage)
    face_encoding = np.array([shape.part(i).x for i in range(68)] + [shape.part(i).y for i in range(68)])
    return face_encoding

# Fonction pour enregistrer un utilisateur et son image dans la base de données
def register_user(username, email, password, image):
    # Vérifier si l'utilisateur ou l'email existe déjà
    if check_user_exists(username, email):
        st.error("Erreur : Utilisateur ou email déjà existants.")
        return

    # Hachage du mot de passe pour la sécurité
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Extraire l'encodage facial pour l'image
    face_encoding = get_face_encoding(image)
    if face_encoding is None:
        return

    # Connexion à la base de données
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO users (username, email, password, face_encoding) VALUES (?, ?, ?, ?)", 
                       (username, email, hashed_password.decode('utf-8'), face_encoding.tobytes()))
        conn.commit()  # Commit les modifications
        st.success(f"Utilisateur {username} inscrit avec succès.")
    except sqlite3.IntegrityError as e:
        st.error(f"Erreur lors de l'enregistrement dans la base de données : {e}")
    except Exception as e:
        st.error(f"Une erreur inconnue est survenue : {e}")
    finally:
        conn.close()

# Fonction de comparaison des images lors de la connexion
def compare_faces(captured_img, stored_face_encoding_blob, threshold=100):
    # Extraire l'encodage facial de l'image capturée
    captured_face_encoding = get_face_encoding(captured_img)
    if captured_face_encoding is None:
        return False

    # Comparer les encodages faciaux avec un seuil plus tolérant
    distance = np.linalg.norm(captured_face_encoding - np.frombuffer(stored_face_encoding_blob, dtype=np.float64))
    st.write(f"Distance entre les images : {distance}")  # Afficher la distance pour déboguer
    return distance < threshold  # Seuil ajusté pour la comparaison des visages (à tester et ajuster)

# Fonction de connexion via reconnaissance faciale
def login_user(username, password):
    captured_img = capture_single_image()

    if captured_img is None:
        return "Erreur lors de la capture de l'image."

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password, face_encoding FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        stored_password = result[0]
        stored_face_encoding_blob = result[1]

        # Vérifier si le mot de passe est correct
        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):  # Correction du check du mot de passe
            # Comparaison des visages avec un seuil ajusté
            threshold = 5000  # Ajustez ce seuil pour voir ce qui fonctionne le mieux
            if compare_faces(captured_img, stored_face_encoding_blob, threshold):
                st.session_state.logged_in = True  # Flag pour vérifier la connexion réussie
                return f"Connexion réussie ! Bienvenue, {username}."
            else:
                return "La reconnaissance faciale a échoué. Essayez à nouveau."
        else:
            return "Mot de passe incorrect."
    else:
        return "Utilisateur non trouvé."

# Fonction pour traiter l'image téléchargée
def process_uploaded_image(uploaded_file):
    """
    Convertir l'image téléchargée via Streamlit en tableau numpy pour le traitement.
    """
    image = Image.open(uploaded_file)  # Utilisation de PIL pour ouvrir l'image
    image = np.array(image)  # Conversion de l'image en tableau numpy pour OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir de RGB à BGR pour OpenCV
    return image_bgr

# Fonction de recherche d'images (CBIR)
def load_images():
    """
    Charge les images depuis le dossier `datasets/`.
    """
    images = []
    filenames = []
    for filename in os.listdir('Dataset/'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join('Dataset/', filename)
            img = cv2.imread(img_path)
            images.append(img)
            filenames.append(filename)
    return images, filenames

def extract_features(image):
    """
    Extrait les caractéristiques d'une image pour la recherche d'images similaires.
    """
    return np.random.rand(10)  # Exemple simple avec des caractéristiques aléatoires

def calculate_similarity(feature1, feature2, distance_metric='euclidean'):
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
    # Charger les images depuis le dossier
    images, filenames = load_images()

    # Vérifier s'il y a suffisamment d'images
    if len(images) == 0:
        st.error("Aucune image disponible dans le dossier.")
        return

    # Extraction des caractéristiques de l'image à rechercher
    query_features = extract_features(query_img)

    # Calcul des distances avec les autres images
    distances = []
    for img in images:
        features = extract_features(img)
        distance = calculate_similarity(query_features, features, distance_metric)
        distances.append(distance)

    # Trier les indices des images par distance croissante
    sorted_indices = np.argsort(distances)

    # Limiter le nombre de résultats à afficher en fonction du nombre d'images disponibles
    num_results = min(num_results, len(images))

    # Afficher les résultats
    for i in range(num_results):
        st.image(images[sorted_indices[i]], caption=f"Image similaire {i+1}: {filenames[sorted_indices[i]]}")


# Interface Streamlit
def app():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False  # Flag pour vérifier la connexion réussie

    st.title("Système de Connexion avec Reconnaissance Faciale et Recherche d'Images CBIR")
    
    if not st.session_state.logged_in:
        menu = st.sidebar.selectbox("Choisissez une option", ["Inscription", "Connexion"])
    else:
        menu = st.sidebar.selectbox("Choisissez une option", ["Recherche d'Images CBIR", "Déconnexion"])

    if menu == "Inscription":
        st.subheader("Formulaire d'Inscription")
        username = st.text_input("Nom d'utilisateur")
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type='password')
        
        # Bouton pour capturer l'image
        capture_button = st.button("Capturer Image pour Inscription")

        if capture_button:
            if username and email and password:
                captured_image = capture_single_image()  # Capture l'image
                if captured_image is not None:
                    # Enregistrer l'utilisateur et son image dans la base de données
                    register_user(username, email, password, captured_image)
            else:
                st.error("Veuillez remplir tous les champs.")

    elif menu == "Connexion":
        st.subheader("Formulaire de Connexion")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type='password')
        
        if st.button("Se connecter"):
            if username and password:
                result = login_user(username, password)
                st.write(result)
                if "Connexion réussie" in result:
                    st.session_state.logged_in = True  # Flag pour vérifier la connexion réussie
                else:
                    st.warning("Identifiants incorrects.")

    elif menu == "Recherche d'Images CBIR" and st.session_state.logged_in:
        st.subheader("Recherche d'Images Basée sur le Contenu (CBIR)")

        uploaded_file = st.file_uploader("Téléversez une image à rechercher", type=["jpg", "png"])
        if uploaded_file is not None:
            query_img = process_uploaded_image(uploaded_file)
            st.image(query_img, caption="Image à rechercher", use_column_width=True)

            descriptor = st.selectbox("Choisir un descripteur", ["glcm", "haralick", "bit"])
            distance_metric = st.selectbox("Choisir la mesure de distance", ["euclidean", "manhattan", "chebyshev", "canberra"])

            num_results = st.slider("Nombre d'images similaires", 1, 10, 5)

            if st.button("Rechercher"):
                show_similar_images(query_img, num_results, distance_metric, descriptor)

    elif menu == "Déconnexion":
        st.session_state.logged_in = False
        st.success("Vous êtes déconnecté.")

if __name__ == "__main__":
    app()
