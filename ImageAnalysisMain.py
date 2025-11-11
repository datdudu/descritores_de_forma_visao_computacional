# ImageAnalysisMain.py
from pathlib import Path

from utils.ImageLoader import load_image
from utils.Binarization import binarize_and_fill
from utils.ContourProcessing import find_main_contour
from utils.ShapeDescriptors import compute_descriptors
from utils.Transformations import generate_transformations, compare_transformations
from utils.Visualization import plot_full_analysis


def analisar_imagem_detalhada(img_path: str):
    print("=" * 80)
    print(f"ANÁLISE DETALHADA DA IMAGEM: {Path(img_path).name}")
    print("=" * 80)

    # 1. carregar
    img_original, img_gray, mean_val = load_image(img_path)
    if img_gray is None:
        return

    # 2. binarizar
    binary, binary_filled = binarize_and_fill(img_gray, mean_val)

    # 3. contorno
    contorno_info = find_main_contour(binary_filled, img_gray)
    if contorno_info is None:
        return

    # 4/5. descritores
    descritores, coords = compute_descriptors(contorno_info, binary_filled, img_gray)

    # 6. transformações
    transformacoes = generate_transformations(img_gray, mean_val)
    distancias_trans = compare_transformations(
        transformacoes,
        contorno_info["img_area"],
        descritores
    )

    # 7. visualização
    plot_full_analysis(
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
    )

    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 80)

    return descritores, distancias_trans


if __name__ == "__main__":
    caminho_imagem = "Kimia99_DB/trainimage1_2.png"
    analisar_imagem_detalhada(caminho_imagem)
