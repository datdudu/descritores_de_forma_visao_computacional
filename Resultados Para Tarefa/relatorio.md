## 🧠 **Relatório — Descritores de Forma: Robustez e Capacidade Discriminativa (Dataset Kimia99)**
**Disciplina:** Visão Computacional
**Curso:** Engenharia de Computação – IFCE (2025.2)
**Professor:** Nivando Bezerra

---

### **1. Introdução**

Esta atividade tem como objetivo consolidar os conceitos de **descritores de forma** aplicados em imagens binárias, explorando duas propriedades fundamentais:
- **Robustez (invariância)** a transformações geométricas como rotação e escala;
- **Capacidade discriminativa**, ou seja, o quanto os descritores conseguem separar formas de classes diferentes.

O experimento foi conduzido utilizando o **dataset Kimia99**, composto por **99 silhuetas 2D** de objetos divididos em **9 classes distintas** (como aviões, mãos, ferramentas, etc.), com **11 formas por classe**.
O conjunto é amplamente utilizado para tarefas de análise e classificação baseadas apenas na geometria das formas, desconsiderando textura, cor ou contexto.

---

### **2. Parte 1 — Robustez dos Descritores**

#### **2.1 Tabela de Distâncias Médias (D̄ᵗ)**

| Transformação | Distância Média (D̄ᵗ) |
|----------------|----------------------:|
| Rotação 45°    | 2.674 |
| Rotação 90°    | 1.181 |
| Rotação 180°   | 0.010 |
| Escala 50%     | 10.953 |

*(Fonte: cálculo automático com 99 imagens do dataset)*

#### **2.2 Discussão — Robustez e Invariância**

O gráfico gerado (Figura 1) mostra claramente que os descritores apresentaram **alta invariância à rotação**, mas **baixa robustez à escala**.

- Com **rotações de 90° e 180°**, as distâncias médias entre os vetores de descritores foram pequenas, indicando que os descritores conseguem representar bem a forma mesmo girada.
- A **rotação de 180°** praticamente não alterou os descritores (D̄ᵗ ≈ 0.01), o que reforça a invariância dessas métricas geométricas.
- Em contraste, a **escala (redução para 50%)** causou uma alta variação (D̄ᵗ ≈ 10.95), revelando que a maioria dos descritores utilizados (ex: perímetro/área, compacidade) depende diretamente de dimensões absolutas.

🔍 **Interpretação:**
  - **Circularidade**, **solidez** e **excentricidade** apresentaram boa estabilidade sob rotação.
  - **Perímetro/Área**, **compacidade** e **extent** foram altamente sensíveis à escala, pois seus valores mudam proporcionalmente às dimensões da imagem.

Isso indica que, para aplicações com transformações geométricas variadas (ex.: reconhecimento independente do tamanho do objeto), é essencial usar **descritores normalizados** ou invariantes a escala, como momentos de Hu ou Fourier descriptors.

#### **Figura 1 — Robustez dos Descritores**
*(Gráfico anexo: “Robustez dos Descritores por Transformação.png”)*

---

### **3. Parte 2 — Capacidade Discriminativa**

#### **3.1 Justificativa da Escolha dos Descritores**

Foram selecionados os descritores **Circularidade** e **Alongamento (aspect ratio)** para análise de capacidade discriminativa entre classes.

- **Circularidade (4πA/P²)** mede o quão próxima uma forma está de um círculo perfeito.
  Valores próximos de 1 indicam formas circulares; valores menores indicam figuras irregulares ou alongadas.
- **Alongamento (w/h)** representa a razão entre largura e altura da bounding box mínima da forma.
  Esse descritor diferencia bem formas verticais, horizontais e mais arredondadas.

A combinação destes dois descritores foi escolhida por fornecer uma distinção intuitiva entre **formas largas e estreitas** e **formas circulares e angulares**, o que potencializa a separação entre as classes.

#### **3.2 Gráfico de Dispersão 2D**

O gráfico de dispersão foi construído com os 99 objetos, plotando **Circularidade (eixo X)** e **Alongamento (eixo Y)**, com **cores diferentes para cada classe** (Figura 2).

#### **Figura 2 — Capacidade Discriminativa (Circularidade vs Alongamento)**
*(Gráfico anexo: “Capacidade Discriminativa (Circularidade vs Alongamento).png”)*

#### **3.3 Análise da Distância Extra-Classe**

A Tabela abaixo mostra as **distâncias euclidianas médias** entre os **centróides** das classes:

| Exemplo de Comparações entre Classes | Distância |
|------------------------------------|-----------:|
| Classe 1 ↔ Classe 9 | 0.103 |
| Classe 1 ↔ Classe 6 | 0.500 |
| Classe 2 ↔ Classe 8 | 0.043 |
| Classe 6 ↔ Classe 9 | 0.603 |
| Classe 4 ↔ Classe 5 | 0.091 |

🔎 **Análise Visual:**
- As classes **6 e 9**, **1 e 9** e **4 e 5** mostraram **alta proximidade** (baixa distância), indicando **formas visualmente similares** em circularidade e alongamento.
  → Exemplo: formas com curvaturas suaves e sem pontas marcantes acabam ocupando regiões próximas no espaço de atributos.
- Já pares como **Classe 6 ↔ Classe 9 (0.603)** e **Classe 1 ↔ Classe 6 (0.50)** possuem **boa separação**, sugerindo que pertencem a tipos de silhuetas bem distintas (ex: uma longa/estreita e outra mais circular).

💡 **Conclusão Parcial:**
A dupla de descritores escolhida (Circularidade e Alongamento) se mostra eficiente para **diferenciar classes com topologias distintas**, mas **limitada** para formas com proporções semelhantes.
Descritores adicionais, como **momentos de Hu ou Fourier Shape Descriptors**, poderiam melhorar a separabilidade global.

---

### **4. Conclusão**

O experimento com o **dataset Kimia99** evidenciou que:
- Os **descritores geométricos básicos** (área, perímetro, circularidade, etc.) são **altamente invariantes à rotação**, mas **sensíveis à escala**.
- A **circularidade e o alongamento** conseguem diferenciar parte das classes de forma eficaz, especialmente entre objetos circulares e alongados.
- Ainda assim, há **sobreposição entre classes** cujas formas apresentam proporções similares, limitando a separabilidade quando apenas dois descritores são usados.

Em síntese:
- Para **robustez**, é recomendável o uso de **descritores invariantes a escala**;
- Para **discriminação**, a combinação de **múltiplos descritores** pode gerar resultados mais estáveis e granulares.

Essa análise reforça o papel crucial da escolha de **descritores apropriados ao contexto geométrico** e à **invariância desejada**, sendo fundamental em sistemas de reconhecimento e classificação baseados em forma.

---