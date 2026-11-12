import os
import sys
import numpy as np
import nibabel as nib

sys.path.append(os.path.dirname(__file__))
from dataset import gather_pairs, normalize_intensity
from utils import load_config


def preprocess_pairs(pairs, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for i, (img_path, mask_path) in enumerate(pairs):
        fname = os.path.basename(img_path)
        case_id = fname.replace("_0000.nii.gz", "").replace(".nii.gz", "")

        image = nib.load(img_path).get_fdata().astype(np.float32)
        image = normalize_intensity(image)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        np.save(os.path.join(out_dir, f"{case_id}_image.npy"), image)
        np.save(os.path.join(out_dir, f"{case_id}_mask.npy"), mask)

        print(f"[{i + 1}/{len(pairs)}] processed {case_id}")


def main(config_path: str):
    config = load_config(config_path)

    images_dir = os.path.join(config["data_dir"], "imagesTr")
    labels_dir = os.path.join(config["data_dir"], "labelsTr")
    pairs = gather_pairs(images_dir, labels_dir)

    preprocess_pairs(pairs, config["processed_dir"])


if __name__ == "__main__":
    main(config_path="/content/drive/MyDrive/knee-segmentation-cv/configs/config.yaml")
