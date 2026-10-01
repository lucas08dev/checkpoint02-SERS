# Checkpoint 02 — APIs, Energias Renováveis e Aprendizado de Máquina

## Integrantes
- Lucas Silva de Abreu — RM 572321
- Guilherme Reiche — RM 569918
- Nicolas Nishi — RM 572242
- João Camperlingo - RM 568957
- Enzo Guislandi - RM 569885

## Objetivo
Aplicar aprendizado de máquina a dados públicos de energia renovável em duas tarefas independentes, comparando **três algoritmos em cada uma**:

1. **Classificação:** prever a fonte de um empreendimento de geração (Solar, Eólica ou Hidráulica) a partir da potência outorgada e da localização.
2. **Regressão:** estimar a radiação solar horária em Petrolina (PE) a partir de variáveis meteorológicas e da hora do dia.

As duas tarefas foram feitas em **Python (Google Colab)** e reproduzidas no **Orange Data Mining**.

## Dados

| Tarefa | Fonte | Período / recorte | Arquivo |
|---|---|---|---|
| Classificação | [ANEEL — SIGA](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel) | Até 1.200 registros por tipo de geração (UFV, EOL, UHE, PCH, CGH) | `aneel_classificacao_orange.csv` |
| Regressão | [Open-Meteo — API histórica](https://open-meteo.com/en/docs/historical-weather-api) | Petrolina (PE), −9,39 / −40,50, de 01/04/2025 a 30/06/2025, das 7h às 17h (America/Recife) | `meteo_regressao_orange.csv` |

**Observação:** a API da ANEEL apresentou instabilidade (timeout) durante a execução. Conforme orientação do professor, usamos os CSVs fornecidos no repositório da disciplina, carregados diretamente do GitHub no notebook. As células de consulta às APIs foram mantidas no notebook como referência, mas estão comentadas.

## Estrutura do repositório

```
├── README.md
├── Aula_APIs_Energia_Renovavel_ML.ipynb      # notebook completo
├── aneel_classificacao_orange.csv            # dados da classificação
├── meteo_regressao_orange.csv                # dados da regressão
├── meteo_treino.csv / meteo_teste.csv        # divisão temporal usada no Orange
├── Fluxos_para_Classificacao_e_Regressao.ows # fluxo do Orange
└── imagens/                                  # capturas do Orange
```

## Como executar

1. Abra o notebook `Aula_APIs_Energia_Renovavel_ML.ipynb` no [Google Colab](https://colab.research.google.com) (**Arquivo → Fazer upload de notebook**).
2. Execute **Ambiente de execução → Executar tudo**. Os CSVs são lidos diretamente do GitHub, sem necessidade de upload.
3. Para rodar localmente: `pip install pandas numpy matplotlib scikit-learn` e execute as células em ordem no Jupyter ou no VS Code.
4. Para o Orange: abra o arquivo `.ows` (**File → Open**) e, se necessário, selecione novamente os CSVs nos widgets **File**.

---

## Tarefa 1 — Classificação da fonte (ANEEL)

- **Entradas (X):** `potencia_kw`, `latitude`, `longitude`. **Alvo (y):** `fonte`.
- **Limpeza:** removemos 47 linhas com latitude e longitude iguais a 0 (coordenada ausente), resultando em 3.829 empreendimentos.
- **Avaliação (Python):** divisão estratificada 80%/20%, `random_state=42`. Precision, Recall e F1 com **média macro**.
- Padronização (`StandardScaler`) dentro de um pipeline, ajustada somente no treino, para a Regressão Logística e o KNN.

### Resultados — Python

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Regressão Logística | 0,830 | 0,837 | 0,829 | 0,826 |
| KNN (k=5) | 0,960 | 0,960 | 0,959 | 0,959 |
| **Random Forest (300 árvores)** | **0,977** | **0,977** | **0,976** | **0,977** |

### Resultados — Orange
Avaliação: **Random sampling** estratificado, 80% treino, **10 repetições** (média). O Orange usa média **ponderada** em Precision, Recall e F1.

| Modelo | CA (Accuracy) | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0,803 | 0,810 | 0,803 | 0,801 |
| kNN | 0,865 | 0,867 | 0,865 | 0,866 |
| **Random Forest** | **0,970** | **0,970** | **0,970** | **0,970** |

![Fluxo do Orange](imagens/orange_fluxo.png)
![Test and Score — Classificação](imagens/orange_classificacao_test_score.png)
![Matrizes de confusão](imagens/orange_confusion_matrix.png)

### Conclusões — Classificação
- O **Random Forest** foi o melhor modelo nas duas ferramentas, com cerca de 97% de acurácia.
- A **Regressão Logística** confunde principalmente **Solar com Eólica**: as duas fontes aparecem nas mesmas regiões (sobretudo no Nordeste) e têm faixas de potência parecidas, e um modelo linear não consegue separá-las.
- O **kNN** teve desempenho bem menor no Orange (0,865) do que no Python (0,960). No Python os dados foram padronizados; no Orange, sem padronização, a `potencia_kw` (que chega a milhões de kW) domina o cálculo de distância. Isso mostra na prática a importância da padronização em modelos baseados em distância.
- A acurácia alta deve ser vista com cautela: usinas do mesmo tipo ficam muito próximas umas das outras e há linhas duplicadas, então o modelo pode estar "decorando" regiões.
- **Limitações:** potência e localização não descrevem a tecnologia. Uma usina solar e uma eólica de mesma potência podem estar na mesma cidade. A potência outorgada não é energia gerada, e a quantidade de exemplos por classe **não representa a participação das fontes na matriz energética brasileira**, pois a consulta foi limitada por tipo.

---

## Tarefa 2 — Regressão da radiação solar (Open-Meteo)

- **Entradas (X):** `temperatura_c`, `umidade_pct`, `nuvens_pct`, `vento_kmh`, `hora`. **Alvo (y):** `radiacao_w_m2`. `data_hora` usada apenas para ordenar.
- **Divisão temporal (sem embaralhar):** treino com as primeiras 80% das horas (01/04/2025 07h a 12/06/2025 14h, 800 registros) e teste com as últimas 20% (12/06/2025 15h a 30/06/2025 17h, 201 registros).
- No Orange, a mesma divisão foi reproduzida com dois arquivos (`meteo_treino.csv` e `meteo_teste.csv`) e a opção **Test on test data**.

### Resultados — Python

| Modelo | MAE (W/m²) | MSE ((W/m²)²) | R² |
|---|---|---|---|
| Regressão Linear | 145,2 | 30.034 | 0,360 |
| Árvore de Decisão (profundidade 6) | 90,9 | 15.127 | 0,678 |
| **Random Forest (300 árvores)** | **66,4** | **7.210** | **0,846** |

### Resultados — Orange (Test on test data)

| Modelo | MAE (W/m²) | MSE ((W/m²)²) | RMSE (W/m²) | R² |
|---|---|---|---|---|
| Linear Regression | 145,2 | 30.034 | 173,3 | 0,360 |
| Tree | 86,1 | 13.829 | 117,6 | 0,705 |
| **Random Forest** | **66,3** | **7.294** | **85,4** | **0,845** |

A Regressão Linear teve resultados **idênticos** nas duas ferramentas, o que confirma que a divisão temporal é a mesma. As pequenas diferenças na árvore e no Random Forest vêm das configurações padrão de cada ferramenta.

![Test and Score — Regressão](imagens/orange_regressao_test_score.png)
![Real × Previsto — Random Forest](imagens/orange_scatter_plot.png)

### Conclusões — Regressão
- O **Random Forest** foi o melhor modelo nas duas ferramentas: erro médio de cerca de 66 W/m² e R² ≈ 0,85.
- **Papel da hora:** é a variável mais importante, pois define a altura do Sol. A radiação forma uma **curva em sino** ao longo do dia (cerca de 80 W/m² às 7h, pico de cerca de 730 W/m² ao meio-dia). Como essa relação não é linear, a Regressão Linear vai mal (R² = 0,36 e até previsões negativas), enquanto os modelos baseados em árvores capturam a curva.
- As nuvens e a umidade ajustam a radiação dentro de cada hora, mas sozinhas explicam pouco.
- O período de teste (fim de junho, perto do inverno) tem radiação máxima menor que o treino, o que torna a avaliação mais exigente.
- **Radiação não é geração elétrica:** W/m² mede a potência solar que chega a uma superfície horizontal. A energia gerada (kWh) depende da área, inclinação e orientação dos painéis, da eficiência dos módulos, da temperatura das células, de sujeira e sombreamento e das perdas no inversor. Além disso, os dados do Open-Meteo vêm de modelos e reanálise, não de um sensor instalado em um painel real.

---

## Comparação Python × Orange
| Aspecto | Python | Orange |
|---|---|---|
| Classificação — avaliação | 1 divisão estratificada 80/20 (`random_state=42`) | Random sampling estratificado 80/20, 10 repetições |
| Classificação — média das métricas | Macro | Ponderada (weighted) |
| Classificação — limpeza (0,0) | Removidas 47 linhas | Não removidas |
| Regressão — avaliação | 80% iniciais / 20% finais | Mesma divisão, via arquivos separados |
| Melhor modelo nas duas tarefas | Random Forest | Random Forest |

## Fontes
- ANEEL — SIGA: https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel
- Open-Meteo — Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
