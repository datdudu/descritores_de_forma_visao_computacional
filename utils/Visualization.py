# Visualization.py
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_full_analysis(
    img_path,
    img_original,
    img_gray,
    binary,
    binary_filled,
    contorno_info,
    descritores,
    coords,
    transformacoes,
    distancias_trans
):
    print("\n[7] GERANDO VISUALIZAÇÃO...")

    contorno = contorno_info["contorno"]
    x, y, w, h = contorno_info["bbox"]
    hull = contorno_info["hull"]
    area = contorno_info["area"]
    perimetro = contorno_info["perimetro"]
    img_area = contorno_info["img_area"]

    fig = plt.figure(figsize=(20, 14))

    # 1: original
    ax1 = plt.subplot(4, 5, 1)
    if len(img_original.shape) == 3:
        plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img_original, cmap='gray')
    plt.title('1. Imagem Original', fontweight='bold')
    plt.axis('off')

    # 2: gray
    ax2 = plt.subplot(4, 5, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title('2. Escala de Cinza', fontweight='bold')
    plt.axis('off')

    # 3: bin
    ax3 = plt.subplot(4, 5, 3)
    plt.imshow(binary, cmap='gray')
    plt.title('3. Binarização', fontweight='bold')
    plt.axis('off')

    # 4: filled
    ax3b = plt.subplot(4, 5, 4)
    plt.imshow(binary_filled, cmap='gray')
    plt.title('4. Buracos Preenchidos', fontweight='bold')
    plt.axis('off')

    # 5: contorno
    ax4 = plt.subplot(4, 5, 5)
    img_contorno = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contorno, [contorno], -1, (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(img_contorno, cv2.COLOR_BGR2RGB))
    plt.title('5. Contorno', fontweight='bold')
    plt.axis('off')

    # features
    ax5 = plt.subplot(4, 5, 6)
    img_features = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_features, [contorno], -1, (0, 255, 0), 2)
    cv2.drawContours(img_features, [hull], -1, (255, 0, 0), 2)
    cv2.rectangle(img_features, (x, y), (x+w, y+h), (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB))
    plt.title('Features\n(Verde: Contorno, Azul: Hull, Vermelho: BBox)', fontweight='bold', fontsize=9)
    plt.axis('off')

    # harris
    ax5b = plt.subplot(4, 5, 7)
    plt.imshow(binary_filled, cmap='gray')
    if len(coords) > 0:
        import numpy as np
        plt.plot(coords[:, 1], coords[:, 0], 'ro', markersize=6)
    plt.title(f'Cantos Harris ({len(coords)} pontos)', fontweight='bold', fontsize=9)
    plt.axis('off')

    # transformações
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

    # gráfico de descritores
    ax11 = plt.subplot(4, 5, 16)
    nomes_desc = list(descritores.keys())
    valores_desc = list(descritores.values())
    plt.barh(nomes_desc, valores_desc, color='steelblue')
    plt.xlabel('Valor', fontweight='bold', fontsize=8)
    plt.title('Valores dos Descritores', fontweight='bold', fontsize=9)
    plt.tick_params(axis='both', labelsize=7)
    plt.tight_layout()

    # gráfico distâncias
    ax12 = plt.subplot(4, 5, 17)
    if distancias_trans:
        plt.bar(distancias_trans.keys(), distancias_trans.values(), color='coral')
        plt.xlabel('Transformação', fontweight='bold', fontsize=8)
        plt.ylabel('Distância Euclidiana', fontweight='bold', fontsize=8)
        plt.title('Robustez às Transformações', fontweight='bold', fontsize=9)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.tick_params(axis='y', labelsize=7)

    # tabela
    ax13 = plt.subplot(4, 5, 18)
    ax13.axis('tight')
    ax13.axis('off')
    tabela_data = [[nome, f"{valor:.4f}"] for nome, valor in descritores.items()]
    tabela = ax13.table(
        cellText=tabela_data,
        colLabels=['Descritor', 'Valor'],
        cellLoc='left',
        loc='center',
        colWidths=[0.6, 0.4]
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(7)
    tabela.scale(1, 1.3)
    plt.title('Tabela de Descritores', fontweight='bold', pad=20, fontsize=9)

    # info
    ax14 = plt.subplot(4, 5, 19)
    ax14.axis('off')
    info_text = f"""INFORMAÇÕES DA IMAGEM

Arquivo: {Path(img_path).name}
Dimensões: {img_gray.shape[1]}x{img_gray.shape[0]}
Área: {area:.2f} px²
Perímetro: {perimetro:.2f} px
Contornos: ?
Pontos: {len(contorno)}
Cantos Harris: {len(coords)}
"""
    ax14.text(0.1, 0.5, info_text, fontsize=8, verticalalignment='center',
              family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # legenda
    ax15 = plt.subplot(4, 5, 20)
    ax15.axis('off')
    legenda_text = """INTERPRETAÇÃO

✓ Circularidade ≈ 1: forma circular
✓ Alongamento > 1: forma horizontal
✓ Solidez ≈ 1: sem concavidades
✓ Distância baixa: descritor robusto
"""
    ax15.text(0.1, 0.5, legenda_text, fontsize=7, verticalalignment='center',
              family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f'ANÁLISE COMPLETA: {Path(img_path).name}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    print("✓ Visualização gerada com sucesso!")
