# Transformations.py
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import util, feature

def generate_transformations(img_gray, mean_val):
    print("\n[6] TESTANDO ROBUSTEZ COM TRANSFORMAÇÕES...")
    print("-" * 80)

    transformacoes = {
        'Original': img_gray,
        'Rotação 45°': None,
        'Rotação 90°': None,
        'Rotação 180°': None,
        'Escala 50%': None
    }

    h_img, w_img = img_gray.shape
    centro = (w_img // 2, h_img // 2)
    border_value = 255 if mean_val > 127 else 0

    M_45 = cv2.getRotationMatrix2D(centro, 45, 1.0)
    transformacoes['Rotação 45°'] = cv2.warpAffine(img_gray, M_45, (w_img, h_img), borderValue=border_value)

    M_90 = cv2.getRotationMatrix2D(centro, 90, 1.0)
    transformacoes['Rotação 90°'] = cv2.warpAffine(img_gray, M_90, (w_img, h_img), borderValue=border_value)

    M_180 = cv2.getRotationMatrix2D(centro, 180, 1.0)
    transformacoes['Rotação 180°'] = cv2.warpAffine(img_gray, M_180, (w_img, h_img), borderValue=border_value)

    # escala
    novo_w, novo_h = int(w_img * 0.5), int(h_img * 0.5)
    escalada = cv2.resize(img_gray, (novo_w, novo_h))
    pad_w = (w_img - novo_w) // 2
    pad_h = (h_img - novo_h) // 2
    transformacoes['Escala 50%'] = cv2.copyMakeBorder(
        escalada, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=border_value
    )

    return transformacoes


def compare_transformations(transformacoes, img_area, descritores_base):
    import cv2
    distancias_trans = {}
    vetor_base = np.array(list(descritores_base.values()))

    for nome_trans, img_trans in transformacoes.items():
        if nome_trans == 'Original':
            continue

        mean_trans = np.mean(img_trans)
        if mean_trans > 127:
            _, binary_trans = cv2.threshold(img_trans, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary_trans = cv2.threshold(img_trans, 127, 255, cv2.THRESH_BINARY)

        binary_trans = binary_fill_holes(binary_trans > 0).astype(np.uint8) * 255
        contours_trans, _ = cv2.findContours(binary_trans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_trans) == 0:
            continue

        valid_contours_trans = [
            cnt for cnt in contours_trans
            if 0.01 * img_area < cv2.contourArea(cnt) < 0.95 * img_area
        ]

        if len(valid_contours_trans) == 0:
            contorno_trans = max(contours_trans, key=cv2.contourArea)
        else:
            contorno_trans = max(valid_contours_trans, key=cv2.contourArea)

        area_trans = cv2.contourArea(contorno_trans)
        perimetro_trans = cv2.arcLength(contorno_trans, True)

        # momentos
        M_trans = cv2.moments(contorno_trans)
        desc_trans = {}

        if M_trans['mu20'] + M_trans['mu02'] != 0:
            desc_trans['Excentricidade'] = (
                (M_trans['mu20'] - M_trans['mu02'])**2 + 4*M_trans['mu11']**2
            )**0.5 / (M_trans['mu20'] + M_trans['mu02'])
        else:
            desc_trans['Excentricidade'] = 0

        desc_trans['Circularidade'] = (4 * np.pi * area_trans) / (perimetro_trans ** 2) if perimetro_trans > 0 else 0
        desc_trans['Compacidade'] = (perimetro_trans ** 2) / area_trans if area_trans > 0 else 0
        desc_trans['Razao_P_A'] = perimetro_trans / area_trans if area_trans > 0 else 0

        hull_trans = cv2.convexHull(contorno_trans)
        hull_area_trans = cv2.contourArea(hull_trans)
        desc_trans['Solidez'] = area_trans / hull_area_trans if hull_area_trans > 0 else 0

        x_t, y_t, w_t, h_t = cv2.boundingRect(contorno_trans)
        desc_trans['Alongamento'] = w_t / h_t if h_t > 0 else 0
        rect_area_trans = w_t * h_t
        desc_trans['Extent'] = area_trans / rect_area_trans if rect_area_trans > 0 else 0

        image_float_trans = util.img_as_float(binary_trans)
        harris_response_trans = feature.corner_harris(image_float_trans, k=0.04, sigma=1.5)
        coords_trans = feature.corner_peaks(harris_response_trans, min_distance=5, threshold_rel=0.05)
        desc_trans['Num_Cantos'] = len(coords_trans)

        vetor_trans = np.array(list(desc_trans.values()))
        distancia = np.linalg.norm(vetor_base - vetor_trans)
        distancias_trans[nome_trans] = distancia
        print(f"  • {nome_trans}: Distância = {distancia:.4f}")

    return distancias_trans
