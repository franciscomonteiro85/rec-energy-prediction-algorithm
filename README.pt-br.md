[English](README.md)

# Algoritmo de Previsão de Energia
## Pré-processamento
Para executar os algoritmos disponíveis, primeiro é necessário criar uma pasta para colocar os conjuntos de dados.
Por padrão, o local a ser lido é "../datasets/".
Depois disso, os conjuntos de dados precisam ser melhor organizados e limpos.
O script "organize_by_location.py" deve ser executado para criar um arquivo pickle com os dados.
Este script está na pasta "scripts".
Exemplo: python3 organize_by_location.py ../../datasets/Dataset.xlsx ../data/porto.pkl 1/1/2019

Alguns arquivos pickle estão disponíveis na pasta de dados, fique à vontade para usá-los.

## Execução
Existem vários notebooks neste repositório. Vamos descrever o que cada um faz.

#### Aggregated Porto and Shared
Este notebook cria a soma agregada do consumo de energia de cada casa por intervalo de tempo e executa um modelo de regressão linear e polinomial com diferentes números de registros anteriores.

#### Algorithms Porto, Shared e Banes
Notebooks principais para executar os modelos de aprendizado de máquina. Aqui, o dataframe de características é criado com os passos de tempo anteriores do consumo de energia.
Em seguida, o modelo é treinado deixando 10 casas aleatórias para teste e as restantes para treinamento. Isso é feito 10 vezes, a fim de validar cruzado os dados e, em seguida, fazer uma média para encontrar resultados mais precisos. Além disso, os dados são normalizados entre 0 e 1 para serem comparados com os outros conjuntos de dados. Os modelos de aprendizado de máquina utilizados são Regressão Linear, XGBoost e Random Forest. Por padrão, a saída é escrita em um arquivo .txt na pasta "gpu_logs".

#### Dataset_Energy_Weather_Merging (Não finalizado)
Por enquanto, este notebook apenas resampleia os dados meteorológicos de intervalos de 10 minutos para intervalos de 15 minutos para, em seguida, mesclá-los com o dataset de consumo de energia.

#### Data_Visualization (Não finalizado)
Algumas visualizações dos dados em diferentes períodos de tempo (o ano todo, semanalmente, mensalmente).

#### Graphics
Gráficos comparando o desempenho dos diferentes modelos. Os gráficos criados são salvos na pasta "images".

#### Shared_Dataset_Creation
Agregação das múltiplas casas do dataset compartilhado em um único arquivo.

#### REC_Energy_Algorithm
Primeiros testes no dataset, desatualizados e usando apenas um prédio para treinar o modelo.

#### REC_Without_ML
Testes no dataset Porto para ver se simplesmente atribuir o valor anterior de consumo de energia de 15 minutos à previsão é melhor ou pior do que os modelos. Baseline para comparar com os outros modelos.

## Dados disponíveis
A pasta "data" contém vários arquivos (pickle e csv) dos dados utilizados.
shared_1year.csv contém o dataset compartilhado com dados de consumo de energia de 7 casas ao longo de um ano.
shared_total.csv contém as mesmas informações das casas, mas por um período mais longo (algumas casas apenas alguns meses, outras mais).
porto.pkl contém o dataset do Porto organizado por localização (cada localização é uma casa diferente).
Os pickles das estações são dados meteorológicos de Aveiro e Viseu para usar juntamente com os dados de consumo de energia.

## Requisitos
Esses notebooks são programados para serem executados em uma GPU e, para isso, utilizam algumas bibliotecas. Para executar o modelo XGBoost, é usada a biblioteca Python XGBoost. Para o Random Forest, é usada a biblioteca Rapids-AI cuML, que fornece um modelo semelhante à biblioteca scikit-learn, mas otimizado para GPU.