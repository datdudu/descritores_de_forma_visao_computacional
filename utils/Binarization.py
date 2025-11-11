import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

def binarize_and_fill(img_gray, mean_val):
    print("\n[2] BINARIZAÇÃO...")

    if mean_val > 127:
        print("  - Detectado: Fundo BRANCO, Objeto PRETO")
        _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        print("  - Detectado: Fundo PRETO, Objeto BRANCO")
        _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    binary_filled = binary_fill_holes(binary > 0).astype(np.uint8) * 255
    print("✓ Imagem binarizada e buracos preenchidos")

    return binary, binary_filled
