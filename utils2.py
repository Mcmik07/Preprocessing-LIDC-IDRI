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
