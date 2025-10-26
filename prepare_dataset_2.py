import os
from pathlib import Path
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high

from utils2 import is_dir_path, segment_lung
from pylidc.utils import consensus
import cv2  # <-- OpenCV pour le redimensionnement

warnings.filterwarnings(action='ignore')

# =========================================
# Paramètres de redimensionnement (OpenCV)
# =========================================
OUT_H, OUT_W = 256, 256   # hauteur, largeur cible pour image & masque

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

# Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset', 'LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset', 'MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset', 'IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset', 'META_PATH'))

# Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset', 'Mask_Threshold')

# Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc', 'confidence_level')
padding = parser.getint('pylidc', 'padding_size')


class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMAGE, CLEAN_DIR_MASK,
                 META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding, padding), (padding, padding), (0, 0)]
        self.meta = pd.DataFrame(
            index=[],
            columns=[
                'patient_id', 'nodule_no', 'slice_no',
                'original_image', 'mask_image',
                'malignancy', 'is_cancer', 'is_clean'
            ]
        )

    def calculate_malignancy(self, nodule):
        """
        Calcule la malignité d’un nodule à partir des annotations (1..5)
        Retourne (median_high, True/False/'Ambiguous')
        """
        list_of_malignancy = []
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if malignancy > 3:
            return malignancy, True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'

    def save_meta(self, meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list, index=[
            'patient_id', 'nodule_no', 'slice_no',
            'original_image', 'mask_image',
            'malignancy', 'is_cancer', 'is_clean'
        ])
        self.meta = pd.concat([self.meta, tmp.to_frame().T], ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        for patient in tqdm(self.IDRI_list):
            pid = patient  # LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

            # Garde-fou si aucun scan n’est trouvé
            if scan is None:
                print(f"[WARN] No scan found for {pid}, skipping.")
                continue

            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(
                pid, vol.shape, len(nodules_annotation)
            ))

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    # nodule est un cluster = liste d’annotations
                    diameters = [ann.diameter for ann in nodule if hasattr(ann, 'diameter')]
                    if not diameters:
                        print(f"[WARN] No diameters for patient {pid}, nodule {nodule_idx}, skipping.")
                        continue

                    # Exclure les nodules de taille inférieure à 4 mm (max des annotations)
                    if max(diameters) < 4.0:
                        print(f"Excluding nodule with max diameter {max(diameters):.2f} mm for patient {pid}")
                        continue

                    # Génère masque de consensus et bounding box
                    mask, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                    lung_np_array = vol[cbbox]

                    # Calcul de la malignité
                    malignancy, cancer_label = self.calculate_malignancy(nodule)

                    for nodule_slice in range(mask.shape[2]):
                        # Filtre par taille minimale du masque (en pixels) AVANT redimensionnement
                        if np.sum(mask[:, :, nodule_slice]) <= self.mask_threshold:
                            continue

                        # Segmenter uniquement la partie du poumon
                        slice_img = lung_np_array[:, :, nodule_slice]
                        lung_segmented_np_array = segment_lung(slice_img)

                        # --- Redimensionnement OpenCV ---
                        # Image CT : bilinéaire (INTER_LINEAR)
                        img_resized = cv2.resize(
                            lung_segmented_np_array.astype(np.float32),
                            (OUT_W, OUT_H),  # (width, height)
                            interpolation=cv2.INTER_LINEAR
                        ).astype(lung_segmented_np_array.dtype)

                        # Masque : nearest (INTER_NEAREST) pour rester binaire
                        msk_resized = cv2.resize(
                            mask[:, :, nodule_slice].astype(np.uint8),
                            (OUT_W, OUT_H),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint8)
                        # --------------------------------

                        # Nommage des fichiers pour chaque nodule
                        nodule_name = "{}:{}_NI{}_slice{}".format(pid,pid[-4:], prefix[nodule_idx], prefix[nodule_slice])
                        mask_name = "{}:{}_MA{}_slice{}".format(pid,pid[-4:], prefix[nodule_idx], prefix[nodule_slice])

                        # Liste des métadonnées à sauvegarder
                        meta_list = [
                            pid[-4:],                # patient_id (suffixe)
                            nodule_idx,              # nodule_no
                            prefix[nodule_slice],    # slice_no
                            nodule_name,             # original_image (basename)
                            mask_name,               # mask_image (basename)
                            malignancy,              # malignancy
                            cancer_label,            # is_cancer (peut être True/False/'Ambiguous')
                            False                    # is_clean
                        ]

                        # Sauvegarder les métadonnées et les fichiers (version redimensionnée)
                        self.save_meta(meta_list)
                        np.save(patient_image_dir / nodule_name, img_resized)
                        np.save(patient_mask_dir / mask_name, msk_resized)

            else:
                print("Clean Dataset", pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                # There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                for slice in range(vol.shape[2]):
                    if slice > 50:
                        break
                    lung_segmented_np_array = segment_lung(vol[:, :, slice])
                    lung_mask = np.zeros_like(lung_segmented_np_array, dtype=np.uint8)

                    # --- Redimensionnement OpenCV ---
                    img_resized = cv2.resize(
                        lung_segmented_np_array.astype(np.float32),
                        (OUT_W, OUT_H),
                        interpolation=cv2.INTER_LINEAR
                    ).astype(lung_segmented_np_array.dtype)

                    msk_resized = cv2.resize(
                        lung_mask.astype(np.uint8),
                        (OUT_W, OUT_H),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.uint8)
                    # --------------------------------

                    # CN = CleanNodule, CM = CleanMask
                    nodule_name = "{}:{}_CN001_slice{}".format(pid,pid[-4:], prefix[slice])
                    mask_name = "{}:{}_CM001_slice{}".format(pid,pid[-4:], prefix[slice])
                    meta_list = [pid[-4:], slice, prefix[slice], nodule_name, mask_name, 0, False, True]

                    self.save_meta(meta_list)
                    np.save(patient_clean_dir_image / nodule_name, img_resized)
                    np.save(patient_clean_dir_mask / mask_name, msk_resized)

        print("Saved Meta data")
        self.meta.to_csv(self.meta_path + 'meta_info.csv', index=False)


if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file
    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    test = MakeDataSet(
        LIDC_IDRI_list, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMAGE, CLEAN_DIR_MASK,
        META_DIR, mask_threshold, padding, confidence_level
    )
    test.prepare_dataset()
