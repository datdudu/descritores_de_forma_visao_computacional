import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from skimage import util, feature
from scipy.ndimage import binary_fill_holes

# ============================================
# FUNÇÃO PARA ANÁLISE DETALHADA DE UMA IMAGEM
# ============================================

def analisar_imagem_detalhada(img_path):
    """
    Analisa uma imagem específica mostrando todo o processo detalhado
    """
    print("=" * 80)
    print(f"ANÁLISE DETALHADA DA IMAGEM: {Path(img_path).name}")
    print("=" * 80)
    
    # 1. CARREGAR IMAGEM
    print("\n[1] CARREGANDO IMAGEM...")
    img_original = cv2.imread(str(img_path))
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        print(" Erro ao carregar a imagem!")
        return
    
    print(f"✓ Imagem carregada com sucesso!")
    print(f"  - Dimensões: {img_gray.shape[1]}x{img_gray.shape[0]} pixels")
    print(f"  - Tipo: {img_gray.dtype}")
    print(f"  - Valor médio: {np.mean(img_gray):.2f}")
    
    # 2. BINARIZAÇÃO
    print("\n[2] BINARIZAÇÃO...")
    mean_val = np.mean(img_gray)
    
    if mean_val > 127:
        print("  - Detectado: Fundo BRANCO, Objeto PRETO")
        _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        print("  - Detectado: Fundo PRETO, Objeto BRANCO")
        _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Preencher buracos
    binary_filled = binary_fill_holes(binary > 0).astype(np.uint8) * 255
    print(f"✓ Imagem binarizada e buracos preenchidos")
    
    # 3. DETECÇÃO DE CONTORNOS
    print("\n[3] DETECÇÃO DE CONTORNOS...")
    contours, hierarchy = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"✓ Contornos encontrados: {len(contours)}")
    
    if len(contours) == 0:
        print(" Nenhum contorno encontrado!")
        return
    
    # Filtrar contornos
    img_area = img_gray.shape[0] * img_gray.shape[1]
    print(f"  - Área total da imagem: {img_area} pixels²")
    
    for i, cnt in enumerate(contours):
        area_cnt = cv2.contourArea(cnt)
        percent = (area_cnt / img_area) * 100
        print(f"  - Contorno {i}: {area_cnt:.0f} pixels² ({percent:.1f}% da imagem)")
    
    # Filtrar contornos válidos
    valid_contours = []
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        if 0.01 * img_area < area_cnt < 0.95 * img_area:
            valid_contours.append(cnt)
    
    if len(valid_contours) == 0:
        contorno = max(contours, key=cv2.contourArea)
        print(f"  ⚠ Usando o maior contorno (sem filtro)")
    else:
        contorno = max(valid_contours, key=cv2.contourArea)
        print(f"  ✓ Usando o maior contorno válido")
    
    print(f"  - Contorno selecionado: {len(contorno)} pontos")
    
    # 4. CÁLCULO DE PROPRIEDADES BÁSICAS
    print("\n[4] PROPRIEDADES BÁSICAS...")
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    print(f"  - Área: {area:.2f} pixels²")
    print(f"  - Perímetro: {perimetro:.2f} pixels")
    
    # Bounding Box
    x, y, w, h = cv2.boundingRect(contorno)
    print(f"  - Bounding Box: ({x}, {y}) - Largura: {w}, Altura: {h}")
    
    # Convex Hull
    hull = cv2.convexHull(contorno)
    hull_area = cv2.contourArea(hull)
    print(f"  - Área do Convex Hull: {hull_area:.2f} pixels²")
    
    # 5. CÁLCULO DOS DESCRITORES
    print("\n[5] DESCRITORES DE FORMA...")
    print("-" * 80)
    
    descritores = {}
    
    # Momentos
    M = cv2.moments(contorno)
    
    # 5.1 Excentricidade
    if M['mu20'] + M['mu02'] != 0:
        excentricidade = ((M['mu20'] - M['mu02'])**2 + 4*M['mu11']**2)**0.5 / (M['mu20'] + M['mu02'])
    else:
        excentricidade = 0
    descritores['Excentricidade'] = excentricidade
    print(f"  • Excentricidade: {excentricidade:.4f}")
    print(f"    (Mede o quão alongada é a forma - 0: circular, 1: muito alongada)")
    
    # 5.2 Circularidade
    if perimetro > 0:
        circularidade = (4 * np.pi * area) / (perimetro ** 2)
    else:
        circularidade = 0
    descritores['Circularidade'] = circularidade
    print(f"\n  • Circularidade: {circularidade:.4f}")
    print(f"    (Quão próximo de um círculo - 1: círculo perfeito, <1: menos circular)")
    
    # 5.3 Compacidade
    if area > 0:
        compacidade = (perimetro ** 2) / area
    else:
        compacidade = 0
    descritores['Compacidade'] = compacidade
    print(f"\n  • Compacidade: {compacidade:.4f}")
    print(f"    (Relação perímetro²/área - menor valor = mais compacta)")
    
    # 5.4 Razão Perímetro/Área
    if area > 0:
        razao_pa = perimetro / area
    else:
        razao_pa = 0
    descritores['Razao_P_A'] = razao_pa
    print(f"\n  • Razão Perímetro/Área: {razao_pa:.4f}")
    print(f"    (Complexidade da borda)")
    
    # 5.5 Solidez
    if hull_area > 0:
        solidez = area / hull_area
    else:
        solidez = 0
    descritores['Solidez'] = solidez
    print(f"\n  • Solidez: {solidez:.4f}")
    print(f"    (Proporção da área preenchida - 1: sem concavidades)")
    
    # 5.6 Alongamento
    if h > 0:
        alongamento = w / h
    else:
        alongamento = 0
    descritores['Alongamento'] = alongamento
    print(f"\n  • Alongamento (Aspect Ratio): {alongamento:.4f}")
    print(f"    (Razão largura/altura - 1: quadrado, >1: horizontal, <1: vertical)")
    
    # 5.7 Extent
    rect_area = w * h
    if rect_area > 0:
        extent = area / rect_area
    else:
        extent = 0
    descritores['Extent'] = extent
    print(f"\n  • Extent: {extent:.4f}")
    print(f"    (Proporção da área em relação ao bounding box)")
    
    # 5.8 Número de Cantos usando Harris Corner Detection
    print(f"\n  • Detectando cantos com Harris Corner Detection...")
    image_float = util.img_as_float(binary_filled)
    harris_response = feature.corner_harris(image_float, k=0.04, sigma=1.5)
    coords = feature.corner_peaks(harris_response, min_distance=5, threshold_rel=0.05)
    num_cantos = len(coords)
    descritores['Num_Cantos'] = num_cantos
    print(f"  • Número de Cantos (Harris): {num_cantos}")
    print(f"    (Pontos de curvatura detectados)")
    
    print("-" * 80)
    
    # 6. TRANSFORMAÇÕES E ROBUSTEZ
    print("\n[6] TESTANDO ROBUSTEZ COM TRANSFORMAÇÕES...")
    print("-" * 80)
    
    transformacoes = {
        'Original': img_gray,
        'Rotação 45°': None,
        'Rotação 90°': None,
        'Rotação 180°': None,
        'Escala 50%': None
    }
    
    # Aplicar rotações
    h_img, w_img = img_gray.shape
    centro = (w_img // 2, h_img // 2)
    border_value = 255 if mean_val > 127 else 0
    
    M_45 = cv2.getRotationMatrix2D(centro, 45, 1.0)
    transformacoes['Rotação 45°'] = cv2.warpAffine(img_gray, M_45, (w_img, h_img), borderValue=border_value)
    
    M_90 = cv2.getRotationMatrix2D(centro, 90, 1.0)
    transformacoes['Rotação 90°'] = cv2.warpAffine(img_gray, M_90, (w_img, h_img), borderValue=border_value)
    
    M_180 = cv2.getRotationMatrix2D(centro, 180, 1.0)
    transformacoes['Rotação 180°'] = cv2.warpAffine(img_gray, M_180, (w_img, h_img), borderValue=border_value)
    
    # Aplicar escala
    novo_w, novo_h = int(w_img * 0.5), int(h_img * 0.5)
    escalada = cv2.resize(img_gray, (novo_w, novo_h))
    pad_w = (w_img - novo_w) // 2
    pad_h = (h_img - novo_h) // 2
    transformacoes['Escala 50%'] = cv2.copyMakeBorder(escalada, pad_h, pad_h, pad_w, pad_w, 
                                                       cv2.BORDER_CONSTANT, value=border_value)
    
    # Calcular descritores para cada transformação
    vetor_base = np.array(list(descritores.values()))
    distancias_trans = {}
    
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
        
        if len(contours_trans) > 0:
            # Filtrar contornos
            valid_contours_trans = [cnt for cnt in contours_trans 
                                   if 0.01 * img_area < cv2.contourArea(cnt) < 0.95 * img_area]
            
            if len(valid_contours_trans) == 0:
                contorno_trans = max(contours_trans, key=cv2.contourArea)
            else:
                contorno_trans = max(valid_contours_trans, key=cv2.contourArea)
            
            area_trans = cv2.contourArea(contorno_trans)
            perimetro_trans = cv2.arcLength(contorno_trans, True)
            
            # Calcular descritores
            M_trans = cv2.moments(contorno_trans)
            desc_trans = {}
            
            # Excentricidade
            if M_trans['mu20'] + M_trans['mu02'] != 0:
                desc_trans['Excentricidade'] = ((M_trans['mu20'] - M_trans['mu02'])**2 + 4*M_trans['mu11']**2)**0.5 / (M_trans['mu20'] + M_trans['mu02'])
            else:
                desc_trans['Excentricidade'] = 0
            
            # Circularidade
            desc_trans['Circularidade'] = (4 * np.pi * area_trans) / (perimetro_trans ** 2) if perimetro_trans > 0 else 0
            
            # Compacidade
            desc_trans['Compacidade'] = (perimetro_trans ** 2) / area_trans if area_trans > 0 else 0
            
            # Razão P/A
            desc_trans['Razao_P_A'] = perimetro_trans / area_trans if area_trans > 0 else 0
            
            # Solidez
            hull_trans = cv2.convexHull(contorno_trans)
            hull_area_trans = cv2.contourArea(hull_trans)
            desc_trans['Solidez'] = area_trans / hull_area_trans if hull_area_trans > 0 else 0
            
            # Alongamento
            x_t, y_t, w_t, h_t = cv2.boundingRect(contorno_trans)
            desc_trans['Alongamento'] = w_t / h_t if h_t > 0 else 0
            
            # Extent
            rect_area_trans = w_t * h_t
            desc_trans['Extent'] = area_trans / rect_area_trans if rect_area_trans > 0 else 0
            
            # Num Cantos (Harris)
            image_float_trans = util.img_as_float(binary_trans)
            harris_response_trans = feature.corner_harris(image_float_trans, k=0.04, sigma=1.5)
            coords_trans = feature.corner_peaks(harris_response_trans, min_distance=5, threshold_rel=0.05)
            desc_trans['Num_Cantos'] = len(coords_trans)
            
            # Calcular distância euclidiana
            vetor_trans = np.array(list(desc_trans.values()))
            distancia = np.linalg.norm(vetor_base - vetor_trans)
            distancias_trans[nome_trans] = distancia
            
            print(f"  • {nome_trans}: Distância = {distancia:.4f}")
    
    print("-" * 80)
    
    # 7. VISUALIZAÇÃO COMPLETA
    print("\n[7] GERANDO VISUALIZAÇÃO...")
    
    fig = plt.figure(figsize=(20, 14))
    
    # Linha 1: Processo de extração
    # Imagem Original
    ax1 = plt.subplot(4, 5, 1)
    if len(img_original.shape) == 3:
        plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img_original, cmap='gray')
    plt.title('1. Imagem Original', fontweight='bold')
    plt.axis('off')
    
    # Imagem em Escala de Cinza
    ax2 = plt.subplot(4, 5, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title('2. Escala de Cinza', fontweight='bold')
    plt.axis('off')
    
    # Imagem Binarizada
    ax3 = plt.subplot(4, 5, 3)
    plt.imshow(binary, cmap='gray')
    plt.title('3. Binarização', fontweight='bold')
    plt.axis('off')
    
    # Imagem com buracos preenchidos
    ax3b = plt.subplot(4, 5, 4)
    plt.imshow(binary_filled, cmap='gray')
    plt.title('4. Buracos Preenchidos', fontweight='bold')
    plt.axis('off')
    
    # Contorno Detectado
    ax4 = plt.subplot(4, 5, 5)
    img_contorno = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contorno, [contorno], -1, (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(img_contorno, cv2.COLOR_BGR2RGB))
    plt.title('5. Contorno', fontweight='bold')
    plt.axis('off')
    
    # Linha 2: Características Geométricas
    # Features
    ax5 = plt.subplot(4, 5, 6)
    img_features = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_features, [contorno], -1, (0, 255, 0), 2)
    cv2.drawContours(img_features, [hull], -1, (255, 0, 0), 2)
    cv2.rectangle(img_features, (x, y), (x+w, y+h), (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB))
    plt.title('Features\n(Verde: Contorno, Azul: Hull, Vermelho: BBox)', fontweight='bold', fontsize=9)
    plt.axis('off')
    
    # Cantos Harris
    ax5b = plt.subplot(4, 5, 7)
    plt.imshow(binary_filled, cmap='gray')
    if len(coords) > 0:
        plt.plot(coords[:, 1], coords[:, 0], 'ro', markersize=6)
    plt.title(f'Cantos Harris ({num_cantos} pontos)', fontweight='bold', fontsize=9)
    plt.axis('off')
    
    # Linha 3: Transformações
    idx = 11
    for nome_trans, img_trans in transformacoes.items():
        ax = plt.subplot(4, 5, idx)
        plt.imshow(img_trans, cmap='gray')
        if nome_trans in distancias_trans:
            plt.title(f'{nome_trans}\nDist: {distancias_trans[nome_trans]:.3f}', fontweight='bold', fontsize=9)
        else:
            plt.title(f'{nome_trans}', fontweight='bold', fontsize=9)
        plt.axis('off')
        idx += 1
    
    # Linha 4: Gráficos de Descritores
    # Gráfico de barras dos descritores
    ax11 = plt.subplot(4, 5, 16)
    nomes_desc = list(descritores.keys())
    valores_desc = list(descritores.values())
    plt.barh(nomes_desc, valores_desc, color='steelblue')
    plt.xlabel('Valor', fontweight='bold', fontsize=8)
    plt.title('Valores dos Descritores', fontweight='bold', fontsize=9)
    plt.tick_params(axis='both', labelsize=7)
    plt.tight_layout()
    
    # Gráfico de distâncias
    ax12 = plt.subplot(4, 5, 17)
    if distancias_trans:
        plt.bar(distancias_trans.keys(), distancias_trans.values(), color='coral')
        plt.xlabel('Transformação', fontweight='bold', fontsize=8)
        plt.ylabel('Distância Euclidiana', fontweight='bold', fontsize=8)
        plt.title('Robustez às Transformações', fontweight='bold', fontsize=9)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.tick_params(axis='y', labelsize=7)
    
    # Tabela de descritores
    ax13 = plt.subplot(4, 5, 18)
    ax13.axis('tight')
    ax13.axis('off')
    tabela_data = [[nome, f"{valor:.4f}"] for nome, valor in descritores.items()]
    tabela = ax13.table(cellText=tabela_data, colLabels=['Descritor', 'Valor'],
                        cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(7)
    tabela.scale(1, 1.3)
    plt.title('Tabela de Descritores', fontweight='bold', pad=20, fontsize=9)
    
    # Informações adicionais
    ax14 = plt.subplot(4, 5, 19)
    ax14.axis('off')
    info_text = f"""INFORMAÇÕES DA IMAGEM
    
Arquivo: {Path(img_path).name}
Dimensões: {img_gray.shape[1]}x{img_gray.shape[0]}
Área: {area:.2f} px²
Perímetro: {perimetro:.2f} px
Contornos: {len(contours)}
Pontos: {len(contorno)}
Cantos Harris: {num_cantos}
    """
    ax14.text(0.1, 0.5, info_text, fontsize=8, verticalalignment='center',
              family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legenda de interpretação
    ax15 = plt.subplot(4, 5, 20)
    ax15.axis('off')
    legenda_text = """INTERPRETAÇÃO
    
✓ Circularidade ≈ 1: 
  Forma circular
  
✓ Alongamento > 1: 
  Forma horizontal
  
✓ Solidez ≈ 1: 
  Sem concavidades
  
✓ Distância baixa: 
  Descritor robusto
    """
    ax15.text(0.1, 0.5, legenda_text, fontsize=7, verticalalignment='center',
              family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f'ANÁLISE COMPLETA: {Path(img_path).name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
    print("✓ Visualização gerada com sucesso!")
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 80)
    
    return descritores, distancias_trans


# ============================================
# EXEMPLO DE USO
# ============================================

if __name__ == "__main__":
    # ESCOLHA UMA IMAGEM ESPECÍFICA
    caminho_imagem = "Kimia99_DB/trainimage1_1.png"
    
    # Executar análise detalhada
    descritores, distancias = analisar_imagem_detalhada(caminho_imagem)