#!/usr/bin/env python
# coding: utf-8

#                                     Regressão Logistica
# -> A regressão logistica é um modelo de Machine Learning usado para problemas
# de classificação e não de regressão. Apesar do nome "regressao", a ideia
# principal dela é prever a probabilidade de um resultado pertencer a uma de 
# duas categorias
# 
# -> Analogia: E-mail de Spam: Pense em um problema simples, você quer classificar um e-mail como "spam" ou "não spam". A regressão linear não
# funcionaria bem aqui, pois ela prevê um valor continuo (como o preço de
# uma casa), e você precisa de uma resposta binária. (sim ou não).
# 
# A regressão logistica faz o seguinte:
# 
# -> Ela análisa as caracteristicas do email (quantidade de palavras de spam, remetente, etc)
# 
# -> Em vez de dar uma resposta direta como 0 ou 1, ela calcula a probabilidade,
# um número entre 0 e 1. Por exemplo, ela pode dizer que a probabilidade de um
# e-mail ser spam é de 0.95 (95%)
# 
# -> Você então define um limiar (geralmente 0.5 ou 50%). Se a probabilidade
# calculada for maior que o limiar, o modelo classifica o e-mail como spam.
# Se for menor, ele classifica como não spam.
# 
# -> A regressão logistica usa uma função matemática especial (chamada sigmoide)
# que trasforma o resultado continuo da regressão linear em uma curva em formato
# "S", que limita valores entre 0 e 1.
# 
# -> Resumindo: Calcula a probabilidade de um evento acontecer e usa um limiar para tomar uma decisão de classificação.
# 

#                                  Treinando os dados

# In[18]:


# Função que irá treinar o modelo para classificação dos valores.
# A funão recebe como argumento:
# xtrain: Dados de treino do conjunto x
# ytrain: Dados de treino do conjunto y
def Treinando_o_modelo(xtrain, ytrain):
    
    # Import da classe LogisticRegression que da biblioteca linear_model
    # que tem como objetivo classificar os dados usando a regressão logistica 
    from sklearn.linear_model import LogisticRegression
    
    # Instância da (criação do objeto) classe LogisticRegression. O construtor
    # irá receber como argumento o solver de valor 'lbfgs' que é a maneira
    # que a regressão utilizara para realizar a classificação
    classificador = LogisticRegression(solver='lbfgs')
    
    # Treinando o modelo usando os dados de treino e o método fit
    # que irá aprender a classificar os dados.
    # xtrain[0]: Ao acessar o indiece 0 iremos garantir que o modelo
    # considere apenas os dados do arquivo csv sem usar metadados.
    classificador.fit(xtrain[0], ytrain)
    
    # Retorno do modelo treinado e pronto para realizar predições
    return classificador


#                         Criando a classificação do modelo

# In[19]:


# Função que irá realizar a classificação do modelo. A função irá receber
# como argumento:
# classificador: Ira receber o modelo treinado e pronto para realização da
# classificação.
# xtest: Valores que serão utilizados na classificação
def predictmodel(classificador, xtest):
    
    # Retorno da classificação do modelo
    return classificador.predict(xtest[0])


#                          Criando a matriz de confusão

# In[20]:


# Matriz que irá mostrar a quantidade de valores acertados e errados pelo 
# modelo. A função recebe como argumento:
# classificador: Ira receber o modelo de classificação.

# ytest: Contém os valores reais das classificações
# ira conter a classificação feita pelo modelo.
# Com esses dados poderemos ver a quantidade de acertos
# do modelo através da comparação com os valores reais (ytest)

# ypredição: Classificação do modelo
def matriz_confusao(classificador, ytest, ypredicao):
    
    # Matriz de confusão: É uma tabela que serve para avaliar o desempenho
    # de um modelo de classificação. Ela mostra visualmente o quão bem
    # o seu modelo acertou ou errou as previsões, comparando as previsões
    # do modelo com os valores reais da sua base de dados.
    
    # A matriz tem quatro partes principais, que respondem a perguntas
    # cruciais sobre o seu modelo:
    
    # verdadeiro positivo: O modelo previu que a resposta era 1 e a
    # resposta real era 1. O modelo acertou a previsão positiva.
    
    # verdadeiro negativo: O modelo previu que a resposta era 0 e a
    # resposta real era 0. O modelo acertou a previsão negativa.
    
    # Falso Positivo: O modelo previu que a resposta era 1, mas a 
    # resposta era 0. O modelo errou, e previu positivo onde era 
    # negativo
    
    # Import da classe confusion_matrox da biblioteca metrics do
    # sklearn que tem como objetivo construir matrizes de confusão.
    from sklearn.metrics import confusion_matrix
    
    # Instancia da classe (criação do objeto) confusion_matrix.
    # O construtor irá receber como argumento:
    # ytest: valores reais da classificação.
    # ypredução: Classificação do modelo.
    matriz = confusion_matrix(ytest, ypredicao)

    # Retorno da matriz de confusão
    return matriz


#                 Aplicando o modelo de regressão logistica

# In[21]:


# Import do nosso arquivo de funções
from minhasfuncoes import funcoes

# Método que irá aplicar o modelo de classificação e utilizar
# matriz de confusão. A função recebe como argumento:

# nome_do_arquivo: Local que a base de dados está armazenada.

# delimitador: sinal que separa os dados na base de dados. O
# argumento irá receber como valor padrão a virgula (sinal padrão
# dos arquivos csv).
def computarRegressaoLogistica(nome_do_arquivo, delimitador = ','):
    
    # Trecho que irá carregar o dataset: Como no nosso arquivo de
    # funções nós dividimos os dados em x (caracteristicas) e y
    # (variável alvo) usando o values do pandas (que transforma
    # x e y em um array numpy).
    
    # x: Ira receber o conjunto de caracteristicas.
    
    # y: Irá receber a variável alvo.
    
    # carregar_dataset: Função que irá carregar o dataset na 
    # memória usando o nome do arquivo e o delimitador. 
    x, y = funcoes.carregar_Dataset(nome_do_arquivo, delimitador)
    
    # Ira preencher os dados faltantes da base de dados.
    
    # x: Conjunto que terá os dados preenchidos
    
    # 2: Primeira coluna que irá ter os dados preenchidos (coluna de idade)
    
    # 3: ultima coluna que terá os valores preenchidos (coluna idade).
    x = funcoes.preencherDadosFaltantes(x, 2, 2)
    
    # Como a nossa função de rotulação só pode ser aplicada em apenas
    # uma coluna por vez, vamos utilizar um for que irá aplicar essa
    # função em todas as colunas que queremos rotular.
    
    # Lista que irá conter o indice das colunas categóricas que
    # serão rotuladas. É importante lembrar que como a função
    # rotulação ira apagar a coluna categórica (para inserir de volta
    # a mesma coluna só que rotulada), temos que inverter a ordem dos
    # indices na lista dessa maneira a função conseguira achar o indice
    # da coluna que deve ser convertida
    colunas_para_rotular = [1, 0]
    
    # For que irá percorrer a lista de indices com o objetivo de 
    # acessar cada coluna categórica.
    for i in colunas_para_rotular:
        
        # Aplicando a função de rotulação nas colunas categóricas
        # presentes na lista. A função recebe como argumento 
        # x: Conjunto que contém as colunas que serão convertidas
        # i: Indice das colunas que serão convertidas (indices prese
        # ntes na lista).
        x = funcoes.rotulacao(x, i)
    
    # Ira dividir os dados em treino e teste
    # xtrain: Ira conter as caracteristicas que o modelo usará para
    # aprender a prever os dados.
    
    # ytrain: Irá conter uma parte das classificações que irá ensinar
    # o modelo como ele deve classificar os dados.
    
    # xtest: Irá conter conjunto de dados que o modelo usará para classificar os dados
    
    # y_test: Irá conter os Valores reais da classificação que serão comparados com a predição do modelo
    
    # treino_teste: Função do arquivo funcoes.py que tem como objetivo
    # dividir os dados em treino e teste. A função recebe como argumento:
    
    # x: Irá conter o conjunto de caracteristicas
    
    # y: Irá conter a variável alvo
    
    # 0.2: Tamanho do conjunto de testes (80% para treino, 20% para testes)
    xtrain, xtest, ytrain, ytest = funcoes.treino_teste(x, y, 0.2)
    
    # Chamada da nossa função de normalização: Que tem como objetivo criar
    # escalas (média zero e desvio padrão 1). Dessa maneira conseguimos
    # construir um padrão entre os valores. Iremos aplicar a função nas
    # caracteristicas de treino e teste
    xtrain = funcoes.normalizacao(xtrain)
    
    xtest = funcoes.normalizacao(xtest)
    
    # Ira receber o modelo treinado e pronto para realização
    # da classificação dos dados.
    classificador = Treinando_o_modelo(xtrain, ytrain)
    
    # Ira receber a função que realiza a predição do modelo. A função
    # irá receber como argumento o modelo treinado e as caracteristicas
    # necessárias para realizar a predição.
    ypredicao = predictmodel(classificador, xtest)
    
    # Retono da função que irá construir a matriz confusão. A função irá 
    # receber como argumento:
    
    # Classificador: irá receber o modelo treinado.
    # ytest: Irá conter os valores reais da classificação
    # ypredicao: Ira conter a classificação do modelo.
    return matriz_confusao(classificador, ytest, ypredicao)


# Função que irá mostrar a acurácia (eficiência do modelo)

# In[22]:


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
