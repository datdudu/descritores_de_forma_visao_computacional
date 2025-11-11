# ShapeDescriptors.py
import numpy as np
from skimage import util, feature

def compute_descriptors(contorno_info, binary_filled, img_gray):
    print("\n[5] DESCRITORES DE FORMA...")
    print("-" * 80)

    contorno = contorno_info["contorno"]
    area = contorno_info["area"]
    perimetro = contorno_info["perimetro"]
    x, y, w, h = contorno_info["bbox"]
    hull = contorno_info["hull"]
    hull_area = contorno_info["hull_area"]

    descritores = {}

    # Momentos
    import cv2
    M = cv2.moments(contorno)

    # 5.1 Excentricidade
    if M['mu20'] + M['mu02'] != 0:
        excentricidade = ((M['mu20'] - M['mu02'])**2 + 4*M['mu11']**2)**0.5 / (M['mu20'] + M['mu02'])
    else:
        excentricidade = 0
    descritores['Excentricidade'] = excentricidade
    print(f"  • Excentricidade: {excentricidade:.4f}")

    # 5.2 Circularidade
    if perimetro > 0:
        circularidade = (4 * np.pi * area) / (perimetro ** 2)
    else:
        circularidade = 0
    descritores['Circularidade'] = circularidade
    print(f"\n  • Circularidade: {circularidade:.4f}")

    # 5.3 Compacidade
    if area > 0:
        compacidade = (perimetro ** 2) / area
    else:
        compacidade = 0
    descritores['Compacidade'] = compacidade
    print(f"\n  • Compacidade: {compacidade:.4f}")

    # 5.4 Razão P/A
    if area > 0:
        razao_pa = perimetro / area
    else:
        razao_pa = 0
    descritores['Razao_P_A'] = razao_pa
    print(f"\n  • Razão Perímetro/Área: {razao_pa:.4f}")

    # 5.5 Solidez
    if hull_area > 0:
        solidez = area / hull_area
    else:
        solidez = 0
    descritores['Solidez'] = solidez
    print(f"\n  • Solidez: {solidez:.4f}")

    # 5.6 Alongamento
    if h > 0:
        alongamento = w / h
    else:
        alongamento = 0
    descritores['Alongamento'] = alongamento
    print(f"\n  • Alongamento (Aspect Ratio): {alongamento:.4f}")

    # 5.7 Extent
    rect_area = w * h
    if rect_area > 0:
        extent = area / rect_area
    else:
        extent = 0
    descritores['Extent'] = extent
    print(f"\n  • Extent: {extent:.4f}")

    # 5.8 Cantos (Harris)
    print(f"\n  • Detectando cantos com Harris Corner Detection...")
    image_float = util.img_as_float(binary_filled)
    harris_response = feature.corner_harris(image_float, k=0.04, sigma=1.5)
    coords = feature.corner_peaks(harris_response, min_distance=5, threshold_rel=0.05)
    num_cantos = len(coords)
    descritores['Num_Cantos'] = num_cantos
    print(f"  • Número de Cantos (Harris): {num_cantos}")

    return descritores, coords
