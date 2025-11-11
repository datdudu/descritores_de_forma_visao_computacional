# Projeto: Análise de Descritores de Forma (Kimia99)

Este projeto implementa, em Python, o fluxo pedido na atividade de **Descritores de Forma** usando o dataset **Kimia99**, incluindo: pré-processamento, extração de contorno, cálculo de descritores, aplicação de transformações geométricas e avaliação de robustez via distância euclidiana entre vetores de descritores. A estrutura foi modularizada em vários arquivos dentro da pasta `utils/`, e o arquivo principal (`ImageAnalysisMain.py`) fica na raiz.  
Base da atividade: :contentReference[oaicite:0]{index=0}

---

## 1. Objetivo

A atividade pede que, para cada forma do Kimia99:

1. a gente calcule um conjunto de **descritores de forma** (excentricidade, circularidade, compacidade, solidez, razão perímetro/área, alongamento, extent e número de cantos);
2. aplique **transformações geométricas** (rotações de 45°, 90°, 180° e escala 50%);
3. recalcule os descritores nas imagens transformadas;
4. compare os vetores original × transformado usando **distância euclidiana** para avaliar **robustez/invariância** do descritor à transformação.  
Tudo isso está descrito na **Parte 1: Robustez dos Descritores** da atividade. :contentReference[oaicite:1]{index=1}

---

## 2. Estrutura de Pastas

Supondo que você organizou assim:

```text
descritores_de_forma_visao_computacional/
├── ImageAnalysisMain.py
└── utils/
    ├── ImageLoader.py
    ├── Binarization.py
    ├── ContourProcessing.py
    ├── ShapeDescriptors.py
    ├── Transformations.py
    └── Visualization.py
````

* **`ImageAnalysisMain.py`**: ponto de entrada, orquestra tudo.
* **`utils/`**: contém os módulos que fazem cada etapa do processamento.

---

## 3. Pré-requisitos

### 3.1. Python e bibliotecas

Você vai precisar de:

* Python 3.9+ (o seu aviso anterior era do 3.9)
* OpenCV
* NumPy
* Matplotlib
* SciPy
* scikit-image
* pandas (só para montar a tabela bonitinha na visualização)

Instale assim:

```bash
python3 -m pip install opencv-python numpy matplotlib scipy scikit-image pandas
```

### 3.2. Dataset

repositório do **Kimia 99**

```text
Kimia99_DB/
    trainimage1_1.png
    trainimage1_2.png
    ...
```

---

## 4. Como rodar

Na raiz do projeto (onde está o `ImageAnalysisMain.py`):

```bash
python3 ImageAnalysisMain.py
```

O script vai:

1. carregar a imagem;
2. binarizar e preencher buracos;
3. detectar o maior contorno válido;
4. calcular todos os descritores;
5. gerar as transformações (45°, 90°, 180°, escala 50%);
6. recalcular os descritores nas transformações e medir a distância euclidiana;
7. abrir uma janela do Matplotlib mostrando **todo o processo** (original, binária, contorno, hull, bounding box, cantos Harris, transformações e gráficos).

No final ele imprime no terminal que a análise foi concluída.

---

## 5. Explicação dos arquivo

### 5.1. `ImageAnalysisMain.py`

* **Função**: é o coordenador do fluxo.
* **O que faz**:

  * importa todos os módulos de `utils/`
  * chama na ordem: carregar → binarizar → contorno → descritores → transformações → comparação → visualização
  * imprime o cabeçalho e o rodapé da análise
* **Relação com a atividade**: aqui está a lógica da “Parte 1” descrita no PDF, só que aplicada a uma imagem. Para cumprir 100% a atividade, basta colocar esse laço em cima das 99 imagens e salvar os resultados. 

---

### 5.2. `utils/ImageLoader.py`

* **Função**: isola o carregamento da imagem.
* Carrega a imagem em BGR (original) e em **tons de cinza** (que é o que usamos para binarizar).
* Calcula o **valor médio** da imagem — isso é usado para decidir se o fundo é claro ou escuro.


---

### 5.3. `utils/Binarization.py`

* **Função**: binariza e preenche buracos.
* Usa `cv2.threshold(...)` com inversão dependendo do fundo.
* Usa `binary_fill_holes(...)` para garantir que o objeto fique sólido.
* **Relação com a atividade**: o pré-processamento é o “Passo 1” da Parte 1 (“Pré-processamento e cálculo dos descritores base”). Sem uma segmentação limpa o contorno e os descritores ficam errados. 

---

### 5.4. `utils/ContourProcessing.py`

* **Função**: encontrar o **contorno principal** e calcular as propriedades geométricas básicas.
* Lista todos os contornos, imprime a área de cada um e filtra os que são muito pequenos ou quase do tamanho da imagem.
* Seleciona o **maior contorno válido**.
* Calcula:

  * área do contorno
  * perímetro
  * bounding box (x, y, w, h)
  * convex hull e área do hull
* **Relação com a atividade**: quase todos os descritores de forma clássicos são definidos **sobre o contorno ou sobre a região binária** da forma; então esse passo prepara os dados de entrada para os descritores. É exatamente o que a atividade quer quando fala em “para cada forma, calcule um conjunto de descritores de forma”. 

---

### 5.5. `utils/ShapeDescriptors.py`

* **Função**: calcular o vetor de descritores **Vbase**.
* Calcula exatamente os que a atividade cita como exemplos:

  * Excentricidade
  * Circularidade
  * Compacidade
  * Razão Perímetro/Área
  * Solidez
  * Alongamento (aspect ratio)
  * Extent
  * Número de Cantos (via Harris)
* Guarda tudo em um dicionário Python.
* **Relação com a atividade**: isso é o **Vbase,i** da fórmula de distância que aparece na Parte 1. É esse vetor que vamos comparar com o vetor da imagem transformada. 

---

### 5.6. `utils/Transformations.py`

* **Função**: gerar as imagens transformadas e comparar.
* Cria as 4 transformações pedidas:

  * Rotação 45°
  * Rotação 90°
  * Rotação 180°
  * Escala 50%
* Para **cada** imagem transformada:

  * refaz o binariza → contorno → descritores (mesma lógica do original)
  * monta o vetor **Vtrans**
  * calcula a **distância euclidiana** `||Vtrans - Vbase||₂`
* Imprime no terminal: “Rotação 45°: Distância = ...”
* **Relação com a atividade**: essa parte implementa exatamente o item

  > “Para cada forma (i) e cada transformação (t), calcule a distância Euclidiana (Dt_i) entre o vetor de descritores Base e o vetor de descritores transformados”
  > e mostra, na prática, quais descritores são mais/menos sensíveis à rotação/escala. 

---

### 5.7. `utils/Visualization.py`

* **Função**: mostrar tudo de forma organizada.
* Cria um **painel 4x5** com:

  * imagem original
  * cinza
  * binária
  * binária com buracos preenchidos
  * contorno desenhado
  * contorno + hull + bounding box
  * cantos Harris
  * cada transformação com a distância respectiva
  * gráfico de barras dos descritores
  * gráfico de barras das distâncias (robustez)
  * tabela de descritores
  * texto explicativo
