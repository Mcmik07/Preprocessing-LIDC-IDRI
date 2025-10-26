import os
import numpy as np
import scipy.ndimage as ndimage

def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def segment_lung(img):
    # Fenêtrage de l'image (Windowing)
    img_windowed = np.clip(img, 50, 1000)


    # Normalisation de l'image entre 0 et 1
    img = (img_windowed - np.min(img_windowed)) / (np.max(img_windowed) - np.min(img_windowed))

    return img
"""
import numpy as np

def segment_lung(img):
    # Fenêtrage de l'image (Windowing)
    # Définir les paramètres de fenêtrage
    window_center = 1000  # Centre de la fenêtre (par exemple, pour les poumons)
    window_width = 1500  # Largeur de la fenêtre

    # Calcul des seuils de la fenêtre
    min_hu = window_center - (window_width // 2)
    max_hu = window_center + (window_width // 2)

    # Appliquer le fenêtrage
    img_windowed = np.clip(img, min_hu, max_hu)

    # Normalisation de l'image entre 0 et 1
    # img_normalized = (img_windowed - np.min(img_windowed)) / (np.max(img_windowed) - np.min(img_windowed))

    return img_windowed
"""