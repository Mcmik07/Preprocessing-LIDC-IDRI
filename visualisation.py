import numpy as np
import matplotlib.pyplot as plt

# Charger les fichiers .npy
image_path = "./data/Image/LIDC-IDRI-0001/0001_NI000_slice001.npy"
mask_path = "./data/Mask/LIDC-IDRI-0001/0001_MA000_slice001.npy"

image = np.load(image_path)      # Shape attendue : (H, W)
mask = np.load(mask_path)        # Shape attendue : (H, W) ou (H, W, 1)

# Vérification des formes
print("Image shape :", image.shape)
print("Mask shape  :", mask.shape)

# Si le masque a un canal singleton, on le squeeze
if mask.ndim == 3 and mask.shape[-1] == 1:
    mask = np.squeeze(mask, axis=-1)

# === Affichage ===
plt.figure(figsize=(15, 5))

# Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.axis('off')

# Masque
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Masque')
plt.axis('off')

# Image + Masque superposé
plt.subplot(1, 3, 3)
plt.imshow(image, cmap='gray')
plt.imshow(mask, cmap='jet', alpha=0.4)
plt.title('Image + Masque')
plt.axis('off')

plt.tight_layout()
plt.show()
