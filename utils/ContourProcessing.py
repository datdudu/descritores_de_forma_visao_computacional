# ContourProcessing.py
import cv2

def find_main_contour(binary_filled, img_gray):
    print("\n[3] DETECÇÃO DE CONTORNOS...")
    contours, hierarchy = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"✓ Contornos encontrados: {len(contours)}")

    if len(contours) == 0:
        print(" Nenhum contorno encontrado!")
        return None

    img_area = img_gray.shape[0] * img_gray.shape[1]
    print(f"  - Área total da imagem: {img_area} pixels²")

    # log de áreas
    for i, cnt in enumerate(contours):
        area_cnt = cv2.contourArea(cnt)
        percent = (area_cnt / img_area) * 100
        print(f"  - Contorno {i}: {area_cnt:.0f} pixels² ({percent:.1f}% da imagem)")

    # filtrar contornos válidos
    valid_contours = []
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        if 0.01 * img_area < area_cnt < 0.95 * img_area:
            valid_contours.append(cnt)

    if len(valid_contours) == 0:
        contorno = max(contours, key=cv2.contourArea)
        print("  ⚠ Usando o maior contorno (sem filtro)")
    else:
        contorno = max(valid_contours, key=cv2.contourArea)
        print("  ✓ Usando o maior contorno válido")

    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    x, y, w, h = cv2.boundingRect(contorno)
    hull = cv2.convexHull(contorno)
    hull_area = cv2.contourArea(hull)

    print("\n[4] PROPRIEDADES BÁSICAS...")
    print(f"  - Área: {area:.2f} pixels²")
    print(f"  - Perímetro: {perimetro:.2f} pixels")
    print(f"  - Bounding Box: ({x}, {y}) - Largura: {w}, Altura: {h}")
    print(f"  - Área do Convex Hull: {hull_area:.2f} pixels²")

    return {
        "contorno": contorno,
        "img_area": img_area,
        "area": area,
        "perimetro": perimetro,
        "bbox": (x, y, w, h),
        "hull": hull,
        "hull_area": hull_area
    }
