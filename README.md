# Análise de Mall Customers — Exploração de Modelos Descritivos

Notebook de exploração do conjunto Mall Customers e de técnicas de clusterização, com foco na escolha e avaliação do número de grupos.

## Conteúdo

- `colab_mall_customers_kaggle.ipynb`: exploração inicial, método do cotovelo, K-Means e métricas de validação de clusters.

## Fonte dos dados

O notebook baixa o conjunto [Customer Segmentation Tutorial](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) pelo Kaggle Hub. O CSV não está incluído neste repositório. Acesso à fonte pode exigir conta Kaggle e configuração de autenticação.

## Como executar

1. Instale Python 3.10 ou superior e as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Abra `colab_mall_customers_kaggle.ipynb` no Google Colab ou Jupyter.
3. Execute as células em ordem. A primeira célula baixa os dados com `kagglehub`.

O notebook instala `kagglehub` na primeira célula; em ambiente local, você também pode instalar todas as dependências pelo arquivo `requirements.txt`.

## Escopo

A análise é exploratória e educacional. Os resultados dependem das variáveis escolhidas, da padronização e dos parâmetros do K-Means; as métricas de clusterização devem ser interpretadas em conjunto e não demonstram causalidade.

## Créditos

Notebook de apoio ao módulo de Avaliação de Clusters, com dados publicados por V. J. Choudhary no Kaggle.
