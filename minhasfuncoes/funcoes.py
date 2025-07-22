
# Resolvi criar essa pasta que irá conter as principais funções que iremos
# utilizar durante as aualas, dessa maneira, não precisarei ficar reescrevendo os mesmos blocos de código

 # Ira importar a biblioteca pandas que irá acessar o dataset
 # e manipulará os dados. Também ira conter o get_dummies que
 # usaremos no LabelEncoder
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

# Import da classe LabelEncoder da biblioteca sklearn.preprocessing
# que é usada para transformar rótulos categóricos (como nomes) em
# inteiros (0, 1, 2)
from sklearn.preprocessing import LabelEncoder


# Importa da função train_test_split da biblioteca sklearn.model
# selection que tem como objetivo separar os dados em treino e teste
# para construção de modelos.
from sklearn.model_selection import train_test_split


# Desvio padrão: Medida que indica a distância que os valores estão
# da média, ou seja, quanto mais próximo da média (ou igual a média)
# mais baixo é o desvio padrão, já quanto mais distante, maior é o 
# desvio padrão. A classificação de "desvio padrão alto ou baixo" não considera se o valor individual é maior ou menor que a média.
from sklearn.preprocessing import StandardScaler


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
    
    # Ira retornar os valores de x e y
    return x, y 

# Função que irá preencher dados faltantes da base de dados usando a classe SimpleImputer


# Criação da função que terá como objetivo substituir valores Not a Number
# ou seja linhas que não possuem valores. A função irá receber como parametro
# 2 valores: a variável que terá os valores preenchidos e o tipo de estratégia
# que o SimpleImputer irá utilizar para preencher esses dados como por exemplo a média ou a mediana.
def preencherDadosFaltantes(variavel_x, estrategia):
    # Instância da classe (criação do objeto) SimpleImputer. O construtor
    # da classe irá receber o missing_values que indica o tipo de valor
    # faltante e a estratégia que será utilizada para preenche-los. 
    imputer = SimpleImputer(missing_values=np.nan, strategy=estrategia)
    
    # Irá aplicar os preenchimentos no dataset no intervalo selecionado.
    # O intervalo irá pegar todas as colunas exceto a primeira (que geralm
    # ente contém ids ou nomes).
    variavel_x[:,1:] = imputer.fit_transform(variavel_x[:,1:])
    
    # Retorna da variável x com as alterações
    return variavel_x


# Criação da função que irá transformar dados categóricos em rótulos numéricos
# com o objetivo de conseguir inclui-los nos modelos de predição/classificação

# Criação da função que irá rotular de forma numérica os dados categóricos
# (dados do tipo texto). A função irá receber como paarametro a variável que
# contém os valores categóricos
def rotulacao(variavel_x):
    
    # Instância da classe (criação do objeto) LabelEncoder
    label_encoder_x = LabelEncoder()
    
    # Irá criar e aplicar a rotulação numérica nos valores da primeira
    # coluna (os nomes dos alunos) 
    variavel_x[:, 0] = label_encoder_x.fit_transform(variavel_x[:, 0])
    
    
    # Função que irá aplicar a rotulação binária nos dados categóricos
    binario = pd.get_dummies(variavel_x[:,0])
    
    # Ira usar a função insert da biblioteca numpy para inserir os rótulos
    # binários no conjunto de dados. A função irá receber como parametro:
    # variavel_x: conjunto que irá receber os valores
    # 0: A posição que os dados ficaram no conjunto
    # binario.values: Os valores que serão inseridos
    # axis = 1: A indicação que estamos inserindo uma nova coluna.
    variavel_x = np.insert(variavel_x, 0, binario.values, axis=1)
    
    # Retorni da função com o resultado
    return variavel_x


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

# Função que irá realizar a padronização dos valores do dataset com o objetivo
# de lidar com valores que estão discrepantes
# Função que irá realizar a padronização dos dados com o objetivo
# de definir uma escala de valores que busca evitar a descrepância
# entre os valores. A função ira receber como argumentos os dados de
# treino e teste.
def padronizacao(treino, teste):
    
    # Import da classe StandardScaler da biblioteca skelarn.preprocessing
    # que tem como objetivo aplicar uma escala com média 0  e desvio padrão 1 que irá padronizar o conjunto de valores da base de dados.
    
    # Observação: Vale lembra que a biblioteca preprocessing tem como
    # função preparar os dados antes de utilizarmos como por exemplo
    # na definição de rótulos, na padronização de valores, etc.
    
    # Instância da classe (criação do objeto) Standard
    scale_x = StandardScaler()
    
    # fit_transform da classe standardscaler que irá
    # calcular e aplicar nos dados de treino a padronização
    # dos dados
    treino = scale_x.fit_transform(treino)
    
    # transform da classe StandardScaler que irá aplicar 
    # nos dados de teste a padronização criada nos dados 
    # de treino
    teste = scale_x.transform(teste)
    
    # Retorno das variáveis com os resultados
    return treino, teste


