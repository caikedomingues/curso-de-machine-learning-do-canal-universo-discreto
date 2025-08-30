#!/usr/bin/env python
# coding: utf-8

#                             KNN: K-vizinhos próximos
# 
# -> O processo para classificar um novo ponto de dados é o seguinte:
# 
# -> 1.Escolhe-se um valor para k: k é o número de "vizinhos que o algoritmo
# irá considerar. SE k = 3, ele procurará os 3 vizinhos mais próximos. Se k = 5
# os 5 mais próximos, e assim por diante.
# 
# -> 2.Calcula-se a distância: O algoritmo calcula a distância entre o novo 
# ponto e todos os outros pontos no seu conjunto de dados de treino. A distância euclidiana é a mais comum para isso.
# 
# -> 3.Encontra-se os vizinhos mais próximos: Ele seleciona os k pontos
# que estão mais próximos do novo ponto.
# 
# -> 4.Faz-se uma votação: O modelo verifica a classe ou rótulo de cada desses
# k vizinhos. A classe que aparecer com mais frequência é a vencedora.
# 
# -> 5.Atribui-se a classe: O novo ponto de dados é classificado como
# pertencente á classe vencedora da votação.

# Primeiro, vamos criar a função que irá retornar a classificação do modelo
# knn 

# In[ ]:


# Função que irá criar a predição do modelo. A função tem como argumento:
# classificador: Ira receber a classe/modelo knn treinado e preparado
# para realizar predições

# xtest: Dados que serão utilizados na classificação.
def predicaoModelo(classificador, xtest):
    
    # Retorno da predição do modelo
    return classificador.predict(xtest[0])


# Criação da matriz de confusão que irá mostrar a quantidade de erros e acertos do modelo.

# In[108]:


# Função que irá construir a matriz de confusão. A função irá receber
# como argumento:
# classificador: Modelo knn treinado

# ypredicao: irá conter as predições do modelo knn

# ytest: Valores reias das classificações que serão comparados
# com as predições do modelo knn 
def matriz_confusao(classificador, ypredicao, ytest):
    
    # Import da classe confusion_matrix da biblioteca metrics do módulo
    # sklearn que tem como objetivo criar matrizes de confusão que mostram
    # a quantidade de acertos e erros do modelo.
    from sklearn.metrics import confusion_matrix
    
    # Instância da (criação do objeto) classe confusion_matrix. O 
    # construtor recebe como argumento;
    # ytest: valores reais
    # ypredicao: Predição do modelo
    matriz = confusion_matrix(ytest, ypredicao)
    
    # Retorno da matriz construida com os valores dos argumentos.
    return matriz


# Função que irá construir o modelo KNN

# In[109]:


# Função que irá criar e treinar o modelo usando dados de treino.
# A função recebe como argumento:

# xtrain: Dados de treino de x que irão ensinar o modelo
# a identificar padrões e tendências

# ytrain: Irá conter respostas que ensinam como o modelo deve classificar
# os dados.

def computarKNN(xtrain, ytrain):
    
    # Import da classe KNeighborsClassifier da biblioteca neighbors
    # do módulo sklearn que tem como objetivo criar modelos knn 
    from sklearn.neighbors import KNeighborsClassifier
    
    # Instância da classe KNeighborsClassifier. O construtor irá receber
    # como argumento
    # n_neighbors: Numero de vizinhos que serão procurados na base de 
    # dados. Observação: É importante que o numero de vizinhos seja sempre
    # um valor impar, pois, na votação, sempre devemos ter um grupo que
    # contém um vizinho a mais.
    
    # p: Define o tipo de cálculo que irá calcular a distância entre os 
    # vizinhos. O 2 representa o cálculo de euclides (o mais comum em 
    # modelos knn) 
    classificador = KNeighborsClassifier(n_neighbors=5, p = 2)
    
    # Treinamento do modelo usando os dados de treino
    classificador.fit(xtrain[0], ytrain)
    
    # Retorno do modelo treinado e pronto para realizar predições
    return classificador


# Função que irá aplicar o modelo knn

# In[ ]:


# Import do nosso arquivo de funções
from minhasfuncoes import funcoes

# Criação da função que irá aplicar o knn. A função irá receber como
# argumento:
# nome_arquivo: Local que a base de dados está armazenada.
# delimitador: Sinal que separa os dados no arquivo. O
# argumento terá como valor padrão a ',' que o sinal mais comum
# em arquivos csv.
def knn(nome_arquivo, delimitador=','):
    
    # Trecho que irá carregar o dataset: Como no nosso arquivo de funções
    # nós dividimos os dados do dataset em x (caracteristicas) e y (variável 
    # alvo) da classificação/predição (que resulta transformar x e y em um
    # array numpy), precisamos atribuir a função em 2 variáveis (x e y)
    
    # carregar_Dataset: Função que tem como objetivo carregar o dataset
    # na memória. A função recebe como argumento o nome do arquivo e o
    # delimitador
    x, y = funcoes.carregar_Dataset(nome_arquivo, delimitador)
    
    # Função que irá preencher os dados faltantes usando a média aaritmética.
    # A função recebe como argumento:
    # x: conjunto de dados que será preenchido
    # 2: indice da coluna inicial que será preenchida (coluna de idade)
    # 2: indice da coluna final que será preenchida (coluna de idade)
    x = funcoes.preencherDadosFaltantes(x, 2, 2)
    
    # Agora iremos rotular numericamente os valores categóricos com  o
    # objetivo de possibilitar que o modelo consiga considerar essas
    # colunas em sua predição. Como a nossa função de rotulação aplica
    # os rótulos em apenas uma variável. Dito isso, precisaremos criar um
    # for que irá aplicar a função de rotulação no intervalo de colunas
    # que queremos modificar.
    
    # Lista de indices das colunas que queremos rotular. Vale lembrar que,
    # como os valores categóricos são eliminados (para que os valores 
    # rotulados entre na base de dados), precisamos inverter a ordem dos
    # indices que serão modificados.
    colunas_para_rotular = [1, 0]
    
    # Foe que irá percorrer a lista de indices
    for i in colunas_para_rotular:
        
        # Chamada da função que ira rotular as colunas. Ela ira receber 
        # como argumento:
        # x: Conjunto que será rotulado
        # i: indice da coluna que será rotulada.
        x = funcoes.rotulacao(x, i)
    
    # Trecho que irá separar os dados em conjuntos de treino e teste
    
    # xtrain: Ira conter as caracteristicas que o modelo irá usar para
    # treinar a captura de padrões e tendências.
    
    # ytrain: Ira conter resultados das classificações que irão ensinar
    # como o modelo deve classificar os valores.
    
    # xtest: Caracteristicas que serão utilizadas para realizar a predição
    # do modelo.
    
    # ytest: Valores reais da classificação
    
    # x: conjunto de caracteristicas que serão dividas em treino e teste
    
    # y: Variável alvo que será dividida em treino e teste.
    
    # 0.2: Tamanho do conjunto de testes (20% para testes, 80% para treino)
    xtrain, xtest, ytrain, ytest = funcoes.treino_teste(x, y, 0.2)
    
    # Chamada da função normalização que tem como objetivo criar escalas
    # que padronizam os dados, dessa maneira evitamos um pouco a presençã
    # de outliers. A função recebera como argumentos os conjuntos de treino
    # e teste das caracteristicas.
    xtrain = funcoes.normalizacao(xtrain)
    
    xtest = funcoes.normalizacao(xtest)
    
    # Ira treinar o modelo para realizar a classificação. A
    # função irá receber como argumento oa dados de treino 
    # do modelo.
    classificador = computarKNN(xtrain, ytrain)
    
    # Chamada da função que irá realizar a predição do modelo.
    # A função irá receber como argumento:
    # classificador: Modelo knn treinado.
    
    # xtest: Caracteristicas que serão utilizadas na predição 
    ypredicao = predicaoModelo(classificador, xtest)
    
    # Retorno da matriz de confusão que irá retornar a quantidade
    # de erros e acertos do modelo knn usando o classificador, a
    # predição do modelo e os dados reais que serão comparados
    # com os resultados do modelo.
    return matriz_confusao(classificador, ypredicao, ytest)
