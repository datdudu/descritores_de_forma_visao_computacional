import cv2
import numpy as np
from pathlib import Path

def load_image(img_path: str):
    print("\n[1] CARREGANDO IMAGEM...")
    img_original = cv2.imread(str(img_path))
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        print(" Erro ao carregar a imagem!")
        return None, None, None

    mean_val = float(np.mean(img_gray))

    print(f"✓ Imagem carregada com sucesso!")
    print(f"  - Arquivo: {Path(img_path).name}")
    print(f"  - Dimensões: {img_gray.shape[1]}x{img_gray.shape[0]} pixels")
    print(f"  - Tipo: {img_gray.dtype}")
    print(f"  - Valor médio: {mean_val:.2f}")

    return img_original, img_gray, mean_val
