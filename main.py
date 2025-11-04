import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.spatial import distance
from skimage import util, feature
from scipy.ndimage import binary_fill_holes

# Configuração
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def calcular_descritores(contorno, area, perimetro, binary_img):
    """Calcula diversos descritores de forma"""
    descritores = {}
    
    # Momentos
    M = cv2.moments(contorno)
    
    # Excentricidade
    if M['mu20'] + M['mu02'] != 0:
        excentricidade = ((M['mu20'] - M['mu02'])**2 + 4*M['mu11']**2)**0.5 / (M['mu20'] + M['mu02'])
    else:
        excentricidade = 0
    descritores['Excentricidade'] = excentricidade
    
    # Circularidade
    if perimetro > 0:
        circularidade = (4 * np.pi * area) / (perimetro ** 2)
    else:
        circularidade = 0
    descritores['Circularidade'] = circularidade
    
    # Compacidade
    if area > 0:
        compacidade = (perimetro ** 2) / area
    else:
        compacidade = 0
    descritores['Compacidade'] = compacidade
    
    # Razão Perímetro/Área
    if area > 0:
        razao_pa = perimetro / area
    else:
        razao_pa = 0
    descritores['Razao_P_A'] = razao_pa
    
    # Convex Hull e Solidez
    hull = cv2.convexHull(contorno)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidez = area / hull_area
    else:
        solidez = 0
    descritores['Solidez'] = solidez
    
    # Alongamento (Aspect Ratio)
    x, y, w, h = cv2.boundingRect(contorno)
    if h > 0:
        alongamento = w / h
    else:
        alongamento = 0
    descritores['Alongamento'] = alongamento
    
    # Extent (razão entre área do contorno e área do bounding box)
    rect_area = w * h
    if rect_area > 0:
        extent = area / rect_area
    else:
        extent = 0
    descritores['Extent'] = extent
    
    # Número de cantos usando Harris Corner Detection
    image_float = util.img_as_float(binary_img)
    harris_response = feature.corner_harris(image_float, k=0.04, sigma=1.5)
    coords = feature.corner_peaks(harris_response, min_distance=5, threshold_rel=0.05)
    num_cantos = len(coords)
    descritores['Num_Cantos'] = num_cantos
    
    return descritores

def processar_imagem(img_path):
    """Processa uma imagem e retorna seus descritores"""
    # Carregar imagem
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # Verificar se o fundo é branco ou preto
    mean_val = np.mean(img)
    
    # Se a média for alta, o fundo é branco (objeto preto)
    if mean_val > 127:
        # Binarização normal (objeto fica branco, fundo preto)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        # Binarização invertida (objeto já é branco)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Preencher buracos
    binary_filled = binary_fill_holes(binary > 0).astype(np.uint8) * 255
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Filtrar contornos muito pequenos (ruído) e muito grandes (moldura)
    img_area = img.shape[0] * img.shape[1]
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignorar contornos menores que 1% ou maiores que 95% da imagem
        if 0.01 * img_area < area < 0.95 * img_area:
            valid_contours.append(cnt)
    
    if len(valid_contours) == 0:
        # Se não houver contornos válidos, usar o maior
        contorno = max(contours, key=cv2.contourArea)
    else:
        # Pegar o maior contorno válido
        contorno = max(valid_contours, key=cv2.contourArea)
    
    # Calcular área e perímetro
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    
    # Calcular descritores
    descritores = calcular_descritores(contorno, area, perimetro, binary_filled)
    
    return descritores, img, binary_filled, contorno

def aplicar_rotacao(img, angulo):
    """Aplica rotação na imagem"""
    h, w = img.shape
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    
    # Determinar cor de fundo baseado na média
    mean_val = np.mean(img)
    border_value = 255 if mean_val > 127 else 0
    
    rotacionada = cv2.warpAffine(img, M, (w, h), borderValue=border_value)
    return rotacionada

def aplicar_escala(img, fator):
    """Aplica escala na imagem"""
    h, w = img.shape
    novo_w, novo_h = int(w * fator), int(h * fator)
    escalada = cv2.resize(img, (novo_w, novo_h))
    
    # Determinar cor de fundo
    mean_val = np.mean(img)
    border_value = 255 if mean_val > 127 else 0
    
    # Adicionar padding para manter o tamanho original
    if fator < 1:
        pad_w = (w - novo_w) // 2
        pad_h = (h - novo_h) // 2
        escalada = cv2.copyMakeBorder(escalada, pad_h, pad_h, pad_w, pad_w, 
                                      cv2.BORDER_CONSTANT, value=border_value)
    
    return escalada

# ============================================
# PARTE 1: ROBUSTEZ DOS DESCRITORES
# ============================================

def parte1_robustez(dataset_path):
    """Avalia a robustez dos descritores"""
    print("=" * 60)
    print("PARTE 1: ROBUSTEZ DOS DESCRITORES")
    print("=" * 60)
    
    # Coletar todas as imagens
    imagens = list(Path(dataset_path).rglob("*.png")) + \
              list(Path(dataset_path).rglob("*.jpg")) + \
              list(Path(dataset_path).rglob("*.bmp"))
    
    print(f"\nTotal de imagens encontradas: {len(imagens)}")
    
    # Armazenar descritores base
    descritores_base = []
    imagens_validas = []
    
    print("\n1. Calculando descritores base...")
    for img_path in imagens:
        resultado = processar_imagem(img_path)
        if resultado is not None:
            desc, _, _, _ = resultado
            descritores_base.append(desc)
            imagens_validas.append(img_path)
    
    print(f"   Imagens processadas com sucesso: {len(descritores_base)}")
    
    # Transformações
    transformacoes = {
        'Rotacao_45': lambda img: aplicar_rotacao(img, 45),
        'Rotacao_90': lambda img: aplicar_rotacao(img, 90),
        'Rotacao_180': lambda img: aplicar_rotacao(img, 180),
        'Escala_50': lambda img: aplicar_escala(img, 0.5)
    }
    
    # Calcular distâncias
    distancias = {t: [] for t in transformacoes.keys()}
    
    print("\n2. Aplicando transformações e calculando distâncias...")
    for idx, img_path in enumerate(imagens_validas):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        desc_base = descritores_base[idx]
        vetor_base = np.array(list(desc_base.values()))
        
        for nome_trans, func_trans in transformacoes.items():
            # Aplicar transformação
            img_trans = func_trans(img)
            
            # Processar imagem transformada
            mean_val = np.mean(img_trans)
            if mean_val > 127:
                _, binary_trans = cv2.threshold(img_trans, 127, 255, cv2.THRESH_BINARY_INV)
            else:
                _, binary_trans = cv2.threshold(img_trans, 127, 255, cv2.THRESH_BINARY)
            
            binary_trans = binary_fill_holes(binary_trans > 0).astype(np.uint8) * 255
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary_trans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Filtrar contornos
                img_area = img_trans.shape[0] * img_trans.shape[1]
                valid_contours = [cnt for cnt in contours 
                                 if 0.01 * img_area < cv2.contourArea(cnt) < 0.95 * img_area]
                
                if len(valid_contours) == 0:
                    contorno = max(contours, key=cv2.contourArea)
                else:
                    contorno = max(valid_contours, key=cv2.contourArea)
                
                area = cv2.contourArea(contorno)
                perimetro = cv2.arcLength(contorno, True)
                
                # Calcular descritores transformados
                desc_trans = calcular_descritores(contorno, area, perimetro, binary_trans)
                vetor_trans = np.array(list(desc_trans.values()))
                
                # Calcular distância euclidiana
                dist = np.linalg.norm(vetor_base - vetor_trans)
                distancias[nome_trans].append(dist)
        
        if (idx + 1) % 20 == 0:
            print(f"   Processadas {idx + 1}/{len(imagens_validas)} imagens")
    
    # Calcular distâncias médias
    distancias_medias = {t: np.mean(dists) for t, dists in distancias.items()}
    
    # Criar tabela
    print("\n3. RESULTADOS - Distâncias Médias:")
    print("-" * 60)
    df_distancias = pd.DataFrame({
        'Transformação': list(distancias_medias.keys()),
        'Distância Média (D̄ᵗ)': list(distancias_medias.values())
    })
    print(df_distancias.to_string(index=False))
    print("-" * 60)
    
    # Visualização
    plt.figure(figsize=(10, 6))
    plt.bar(distancias_medias.keys(), distancias_medias.values(), color='steelblue')
    plt.xlabel('Transformação', fontsize=12)
    plt.ylabel('Distância Média', fontsize=12)
    plt.title('Robustez dos Descritores por Transformação', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return descritores_base, imagens_validas, df_distancias

# ============================================
# PARTE 2: CAPACIDADE DISCRIMINATIVA
# ============================================

def parte2_discriminacao(descritores_base, imagens_validas):
    """Avalia a capacidade discriminativa dos descritores"""
    print("\n" + "=" * 60)
    print("PARTE 2: CAPACIDADE DISCRIMINATIVA")
    print("=" * 60)
    
    # Extrair classes dos nomes dos arquivos
    classes = []
    for img_path in imagens_validas:
        # Tentar extrair classe do nome do arquivo ou diretório
        nome_arquivo = img_path.stem  # Nome sem extensão
        # Assumindo padrão: trainimage1_1.png -> classe 1
        if '_' in nome_arquivo:
            classe_num = nome_arquivo.split('_')[0].replace('trainimage', '')
            classes.append(f"Classe_{classe_num}")
        else:
            classe = img_path.parent.name
            classes.append(classe)
    
    # Criar DataFrame
    df = pd.DataFrame(descritores_base)
    df['Classe'] = classes
    
    print(f"\nClasses encontradas: {df['Classe'].nunique()}")
    print(f"Distribuição: \n{df['Classe'].value_counts()}")
    
    # Escolher dois descritores (você pode modificar)
    desc1 = 'Circularidade'
    desc2 = 'Alongamento'
    
    print(f"\n1. Descritores escolhidos: {desc1} e {desc2}")
    print("   Justificativa: Circularidade mede o quão próximo a forma está de um círculo,")
    print("   enquanto Alongamento captura a razão largura/altura, permitindo separar")
    print("   formas arredondadas de alongadas.")
    
    # Gráfico de dispersão
    plt.figure(figsize=(14, 10))
    
    classes_unicas = sorted(df['Classe'].unique())
    cores = plt.cm.tab10(np.linspace(0, 1, len(classes_unicas)))
    
    for idx, classe in enumerate(classes_unicas):
        dados_classe = df[df['Classe'] == classe]
        plt.scatter(dados_classe[desc1], dados_classe[desc2], 
                   label=classe, alpha=0.7, s=100, c=[cores[idx]])
    
    plt.xlabel(desc1, fontsize=12, fontweight='bold')
    plt.ylabel(desc2, fontsize=12, fontweight='bold')
    plt.title(f'Capacidade Discriminativa: {desc1} vs {desc2}', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Análise de separação
    print("\n2. Análise da Separação entre Classes:")
    print("-" * 60)
    
    # Calcular centróides das classes
    centroides = df.groupby('Classe')[[desc1, desc2]].mean()
    
    # Calcular distâncias entre centróides
    print("\nDistâncias entre centróides das classes:")
    for i, classe1 in enumerate(classes_unicas):
        for classe2 in classes_unicas[i+1:]:
            dist = distance.euclidean(centroides.loc[classe1], centroides.loc[classe2])
            print(f"   {classe1} <-> {classe2}: {dist:.4f}")
    
    return df

# ============================================
# EXECUÇÃO PRINCIPAL
# ============================================

if __name__ == "__main__":
    # CONFIGURAR O CAMINHO DO DATASET
    dataset_path = "./Kimia99_DB"  # Ajuste conforme necessário
    
    print("ATIVIDADE: DESCRITORES DE FORMA - KIMIA 99")
    print("IFCE - Engenharia de Computação - 2025.2")
    print()
    
    # Parte 1
    descritores_base, imagens_validas, df_distancias = parte1_robustez(dataset_path)
    
    # Parte 2
    df_resultados = parte2_discriminacao(descritores_base, imagens_validas)
    
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 60)