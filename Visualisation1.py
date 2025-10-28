import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_images_and_masks_for_one_scan(image_root_dir, mask_root_dir, scan_name):
    # Dossier spécifique à visualiser, ici LIDC-IDRI-0001 par exemple
    image_dir = os.path.join(image_root_dir, scan_name)
    mask_dir = os.path.join(mask_root_dir, scan_name)

    # Vérifier que le répertoire existe
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        print(f"Le répertoire {scan_name} n'existe pas.")
        return

    # Afficher le nom du dossier LIDC-IDRI courant
    print(f"Affichage des images et masques pour le scan : {scan_name}")

    # Obtenez les fichiers image et masque dans le répertoire
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Limiter le nombre d'images à afficher
    n_images = min(len(image_files), len(mask_files))

    # Définir la taille de la grille dynamiquement (par exemple, 4 colonnes)
    n_cols = 4
    n_rows = (n_images // n_cols) + (n_images % n_cols > 0)  # Calcul du nombre de lignes nécessaires

    # Affichage pour chaque image et son masque dans ce dossier
    plt.figure(figsize=(15, 5 * n_rows))

    for i in range(n_images):
        # Charger l'image et le masque
        image_path = os.path.join(image_dir, image_files[i])
        mask_path = os.path.join(mask_dir, mask_files[i])

        image = np.load(image_path)  # Shape attendue : (H, W)
        mask = np.load(mask_path)  # Shape attendue : (H, W) ou (H, W, 1)

        # Vérifier si le masque a un canal singleton et l'enlever
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

        # Affichage dans la grille
        plt.subplot(n_rows, n_cols, i + 1)  # Calcul automatique de la grille
        plt.imshow(image, cmap='gray')
        plt.imshow(mask, cmap='jet', alpha=0.5)  # Superposer le masque
        plt.axis('off')

    # Affichage de toutes les images et des masques dans une seule fenêtre
    plt.tight_layout()
    plt.show()  # Attendre la fermeture de la fenêtre avant de passer à l'image suivante


# Dossier racine contenant tous les dossiers LIDC-IDRI
image_root_dir = "./data/Image"
mask_root_dir = "./data/Mask"

# Afficher les images et masques pour un seul scan (par exemple "LIDC-IDRI-0001")
visualize_images_and_masks_for_one_scan(image_root_dir, mask_root_dir, "LIDC-IDRI-0001")
