
# Resolvi criar essa pasta que irá conter as principais funções que iremos
# utilizar durante as aualas, dessa maneira, não precisarei ficar reescrevendo os mesmos blocos de código

 # Ira importar a biblioteca pandas que irá acessar o dataset
 # e manipulará os dados. T
import pandas as pd

# Import da classe SimpleImputer da biblioteca sklearn.impute.
# Essa será a classe que irá preencher os valores que estão faltando
# no dataset. Vale lembrar que a classe SimpleImputer só deve ser uti
# lizada em valores numéricos.
from sklearn.impute import SimpleImputer
    
# Import da biblioteca numpy que irá possibilitar que acessamos
# os valores nan do dataset através do construtor da classe Simple
# Imputer.
import numpy as np 

# Import da classe OneHotEncoder da biblioteca sklear.preprocessing
# que tem como objetivo criar rótulos binários que possibilita que
# o modelo não confunda os rótulos numéricos com dados matemáticos ou
# estatisticos.
from sklearn.preprocessing import OneHotEncoder

# Importa da função train_test_split da biblioteca sklearn.model
# selection que tem como objetivo separar os dados em treino e teste
# para construção de modelos.
from sklearn.model_selection import train_test_split

# Import da classe SimpleImputer da biblioteca sklearn.impute
# que tem como objetivo preencher valores nulos de uma base de
# dados usando estratégias
from sklearn.impute import SimpleImputer
    

# imoort da classe StandardScaler da biblioteca preprocessing
# que tem como objetivo realizar a padronização dos dados.
# Ele faz com que os dados tenham média zero e desvio padrão
# um.
from sklearn.preprocessing import StandardScaler

# Função que tem como objetivo construir barras de progresso em
# estruturas de repetição
from tqdm import tqdm


# Método que irá carregar os arquivos e acessar os valores das colunas. O objetivo da criação desse método é facilitar a leitura dos arquivos csv e facilitar a captura dos valores das colunas, algo que, aparentemente,iremos fazer com frequência nas aulas

# Criação da função que irá carregar o dataset e capturar
# os valores da coluna. A função irá conter 2 parametros:
# O nome do aqruivo que está sendo acessado e o tipo de
# delimitador (caractere que separa os valores).
# Observação: O valor padrão do delimitador será None,
# pois, caso o arquivo tenha como separador a ","(padrão
# dos arquivos csv) não será necessário especificarmos um
# delimitador
def carregar_Dataset(nome_arquivo, delimitador = None):
    
    # Variável que irá acessar o dataset através do seu nome e do
    # seu delimitador
    base_de_dados = pd.read_csv(nome_arquivo, delimiter=delimitador)
    
    # Variável que irá acessar os valores de todas as colunas exceto
    # a ultima (geralmente a variável x representa um conjunto de carac
    # teristicas que serão usadas para treinar modelos de predição/class
    # ificação)
    x = base_de_dados.iloc[:,:-1].values
    
    # Variável que irá conter os valores apenas da ultima coluna. Geralmente
    # a variável y é o alvo da predição/classificação e contém os valores reais
    # da base de dados que serão comparados com a previsão ou classificação do
    # modelo
    y = base_de_dados.iloc[:, -1].values
    
    # Observação:Ao usar a função values,
    # transformaremos as variáveis x e y
    # em arrays numpy
    
    # Ira retornar os valores de x e y
    return x, y 

# Função que irá preencher dados faltantes da base de dados usando a classe SimpleImputer


# Criação da função que terá como objetivo preencher os dados faltantes
# de uma base de dados usando a média aritmética. A função irá receber
# como argumento o conjunto de dados e o intervalo de colunas que
# será preenchido
# x: Conjunto de dados
# inicioColuna: Primeira coluna que sera preenchida
# fimColuna: Ultima coluna a ser preenchida 
def preencherDadosFaltantes(X, inicioColuna, fimColuna):
    
    # Instância da classe SimpleImputer(criação do objeto). O construtor 
    # irá receber como argumento:
    # missing_values: indica o tipo de dado que deve ser preenchido
    # strategy: Indica o tipo de valor que irá preencher os dados, no
    # nosso caso, vamos escolher a média aritmética
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    # Intervalo que será preenchido
    # fit_transform: Ira preencher os dados faltantes com a média
    # definida na estratégia
    X[:,inicioColuna:fimColuna + 1] = imputer.fit_transform(X[:,inicioColuna:fimColuna + 1])
    
    # Retorno dos dados preenchidos.
    return X


# Criação da função que irá rotular de forma binária os dados categóricos
# (dados do tipo texto). A função irá receber como paarametro a variável que contém os valores categóricos e a posição da coluna.
def rotulacao(variavel_x, coluna):
    
    # variavel_x[:,coluna]: Primeiro, esta parte seleciona todos os
    # valores da coluna especificada dentro array variável_x. O ":"
    # significa todas as linhas, e "coluna" especifica o indice da coluna
    # O resultado é um vetor (uma lista unidimensional) de todos os valores daquela coluna.
    
    # Reshape(-1, 1): função reshape() no NumPy é fundamental para manipular a estrutura (forma) de um array sem alterar os dados que ele contém. Ela permite que você reorganize os elementos de um array em uma nova configuração de linhas e colunas (ou dimensões superiores).
    
    # -1: Este é o "coringa" ou "placeholder". Ele significa "calcule esta dimensão automaticamente".

    # 1:Este número fixo (o segundo elemento da tupla (X, Y)) significa que você quer que o array resultante tenha apenas uma coluna. 
    
    # Coluna categorica: variável que terá o resultado armazenado.
    coluna_categorica = variavel_x[:,coluna].reshape(-1, 1)
    
    # Instância da classe onehotencoder: Ela irá receber em seu construtor
    # o sparse_output = False, isso significa que que a saida será será
    # um array com 0s e 1s, o que é mais fácil de manipular e entender.
    # Observação: Por padrão, o OneHotEncoder pode retornar uma matriz esparsa (otimizada para dados com muitos zeros)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    
    # Ira aplicar a rotulação binária na coluna categórica 
    binario = onehot_encoder.fit_transform(coluna_categorica)
    
    # Ira remover a coluna categorica. A função ira receber como
    # parametro:
    # variavel_x: o array original que será removido da coluna categórica
    # coluna: o indice da coluna que terá os valores excluidos
    # axis =1 : A sinalização que estamos excluindo os valores de uma coluna
    remover_valores_coluna_categorica = np.delete(variavel_x, coluna, axis=1)
    
    # Ira concatenar as 2 colunas (a coluna categórica vázia com a coluna
    # coluna que irá conter os valores binários)
    # axis=1: Indica que estamos concatenando colunas
    resultado = np.concatenate((remover_valores_coluna_categorica, binario), axis=1)
    
    # Retorno a concatenação das colunas com os valores binários
    return resultado
    


# Criação da função que irá separar os dados em treino e teste com o objetivo de
# criar modelos de predição/classificação

# Função que irá separa o conjunto de dados em treino e teste.
# A função irá receber como argumento:
# x: Irá conter o conjunto de valores (caracteristicas ) que
# serão utilizadas na classificação/predição do modelo.
# y: Variável alvo da classificação ou predição do modelo
def treino_teste(x, y, tamanho_teste):
      
    # Variáveis que irão receber os valores da função train_test_split
    # com a separação dos dados (em treino e teste).
    # Xtrain: Ira receber a parte de treino dos dados (geralmente a maior
    # parte deles)
    # XTest: Irá conter as características (features) do conjunto de dados de teste.
    # YTrain: Ira conter os valores reais que o modelo usará para aprender
    # YTest: Ira conter os valores reais da predição/classificação
    #que serão comparados com os resultados do modelo
    XTrain, XTest, YTrain, YTest = train_test_split(x, y, test_size= tamanho_teste)
    
    return XTrain, XTest, YTrain, YTest

# Função que irá realizar a normalização dos valores do dataset com o objetivo
# de lidar com valores que estão discrepantes
# Função que irá realizar a padronização dos dados com o objetivo
# de definir uma escala de valores que busca evitar a descrepância
# entre os valores. A função ira receber como argumentos os dados de
# de x (caracteristicas).
def normalizacao(x):

    # instância da classe StandardScaler
    scale = StandardScaler()

    # Ira aprender sobre os dados e aplicar a normalização
    x = scale.fit_transform(x)
    
    # Ira retornar o x normalizado e o objeto standardScaler
    return x, scale



# Função que irá testar a eficiencia de classificação do modelo. A função
# irá receber como argumento a matriz de confusão com os resultados da 
# classificação.
def acuracia(matriz_confusao):
    
    # matriz_confusao[0][0]: Este valor representa os verdadeiros
    # negativos. São as previsões que o modelo classificou como 0
    # e que de fato, eram 0. 
    
    # matriz_confusao[1][1]: Este valor representa os verdadeiros
    # negativos. São as previsões que o modelo classificou como 1
    # e que de fato, eram 1.
    
    # soma (matriz_confusao[0][0] + matriz_confusao[1][1]) no numerador da fórmula representa o total de acertos do modelo.
    
    # matriz_confusao[0][0] + matriz_confusao[1][0] + matriz_confusao[0][1] + matriz_confusao[1][1]: Esta soma representa o total de previsões feitas pelo modelo, ou seja, todos os elementos da matriz de confusão.
    
    accuracy = (matriz_confusao[0][0] + matriz_confusao[1][1]) / (matriz_confusao[0][0] + matriz_confusao[1][0] + matriz_confusao[0][1] + matriz_confusao[1][1])
    return accuracy


# Função que terá como objetivo, testar a eficiência do nossos modelos
# de classificação através de loops que irão executar o modelo várias
# vezes com objetivo de testar o modelo em vários conjuntos diferentes
# (conjuntos de treino e teste).A função irá receber como argumento:

# funcao: Função que irá ser executada dentro do loop.
# tamanho do loop: quantidade de vezes que o loop executará o programa.
# descrição: Irá conter a descrição da barra de progresso. O argumento
# terá como valor padrão a frase 'Não há descrição'.
def testar_modelo_classificacao(funcao, tamanho_do_loop, descricao='Não há descrição'):
    
    # Lista que irá conter as acurácias do modelo após cada execução.
    array_acuracias = []
    
    # Loop for que irá permitir que o modelo seja executado várias vezes.
    # Ele será construido usando a função tqdm que constroi barras de progresso em estruturas de 
    # repetição. A função tqdm recebe como argumento o range com o tamanho do loop e a descrição
    # da barra de progresso.
    for i in tqdm(range(0, tamanho_do_loop), desc=descricao):
        
        # Variável que irá conter os resultados de cada execução
        # do modelo.
        modelo = funcao()
        
        # Inserindo os valores da função acurácia no array de acurácias do
        # modelo. A função recebe como argumento os valores da variável modelo.
        array_acuracias.append(acuracia(modelo))
    
    # Impressão da média dos resultados da acurácia usando a função mean do numpy que 
    # irá retornar a média do conjunto de valores.
    print("Média das acurácias do modelo de classificação: %.2f"% np.mean(array_acuracias))