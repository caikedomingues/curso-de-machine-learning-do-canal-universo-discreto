
# Regressão Linear Multipla: Ocorre quando usamos duas ou mais caracteristicas
# (variáveis independentes) para prever uma única variável alvo (variável dep
# endente).
# 
# Regressão Linear Simples: Ocorre quando utilizamos apenas uma caracteristica
# (variável independente) para prever uma variável alvo (dependente). Por exemplo,
# como fizemos na aula anterior, onde usamos o total de inscritos de um canal
# para prever as suas visualizações.

# Import das bibliotecas necessárias

# In[1]:


# Import da pasta que contém o nosso arquivo de funções
from minhasfuncoes import funcoes

# Biblioteca que contém funções para manipular valores numericos
# e arrays
import numpy as np

# Ira fornecer funções relacionadas ao tempo.
import time

# functools: Fornece ferramentas de ordem superior (funções que operam
# em outras funções) e funções uteis para trabalhar com objetos intocaveis
# (como funções).
# Wraps: É um decorador.
# Decoradores: São uma forma elegante de "embrulhar" ou "modificar"
# funções existentes sem alterar seu código-fonte diretamente. Eles
# adicionam funcionalidades extras (como log, medição de tempo, verificaação
# de permissão) a uma função. Os wraps são importantes por que ao decorar
# uma função python, por padrão, altera metadados importantes da função original
# (como o nome da função, seu docstring, seu módulo). Isso pode causar problemas
# em depuração, documentação ou instropecção de código. O decorador wraps (que
#  você coloca dentro do seu decorador personalizado) preserva esses metadados
# da função original, fazendo com que a função decorada se pareça mais com a
# função original, fazendo com que a função decorada se pareça mais com a função
# original para depuradores e usuários. 
from functools import wraps

# Import do nosso arquivo de funções
from minhasfuncoes import funcoes


# In[ ]:


# Irá treinar os dados para construção do modelo de regressão linear
# multipla. A função irá receber como argumento
# x_train: Dados que irão treinar o modelo com o objetivo de 
# ensina-lo a identificar padrões, tendências e correlações.
# x_test: Ira conter as caracteristicas que o modelo irá usar
# para realizar a predição.
# y_train: Dados de reais que o modelo irá usar para treinar as predições
# y_test: Valores reais que o modelo não tem acesso. Esses valores serão
# comparados com a predição do modelo
def computarRegressaoMultipla(x_train, x_test, y_train, y_test):
    
    # Import da classe LinearRegression da biblioteca sklearm.linear_model
    # que tem como objetivo criar regressões lineares que realizam a predição
    # de valores de uma base de dados
    from sklearn.linear_model import LinearRegression
    
    from sklearn.metrics import r2_score
    
    # Instância da classe de regressão linear
    regressao = LinearRegression()
    
    # Função da classe LinearRegression que irá treinar o modelo
    # usando os dados de treino
    regressao.fit(x_train, y_train)
    
    # Função da classe LinearRegression que ira utilizar o conjunto de 
    # caracteristicas para realizar predições
    predicao = regressao.predict(x_test)
    
    # For que irá percorrer a variável de predição com o objetivo de 
    # imprimir na tela todos os valores das predições feitas pelo modelo
    
    # shape: Serve para mostrar as dimensões do array (quantidade de elementos
    # ou linhas/colunas). No nosso caso, 'predicao' é um vetor (array 1D)
    # onde 'predicao.shape[0]' representa o número total de predições
    # realizadas (que é igual ao número de amostras em x_test)
    # array 1D: Possui apenas 1 linha de valores
    # array 2D: É uma matriz ou tabela que possui linhas e colunas (duas dimensões) 
    for i in range(0, predicao.shape[0]):
        
        # Valores que serão mostrados na tela:
        # predicao: valores previstos pelo modelo
        # y_test: Valores reais da base de dados
        # abs: Ira conter a diferença entre a predição e o valor real
        # com o objetivo de mostrar o quão perto/longe o modelo conseguiu
        # chegar na previsão 
        print(predicao[i], y_test[i], abs(predicao[i] - y_test[i]))
        
        # Irá dar intervalos de 1 segundo na aparição de cada valor
        time.sleep(1)
   
    return r2_score(y_test, predicao)




# In[ ]:


# Irá criar a regressão linear. A função irá receber como parametro:
# nome_arquivo: Nome do arquivo análisado.
# delimitador: O caractere que realiza a separação dos dados no arquivo.
# Esse argumento terá como padrão o None, dessa maneira, caso o arquivo
# contenha a "," como caractere separador, não será necessário passar esse
# valor como argumento.
def regressaoMultipla(nome_arquivo, delimitador = None):
    
    # Ira pegar o segundo atual (inicio da execução do programa)
    tempo_inicial = time.time()
    
    # Ira carregar o dataset que iremos utilizar na construção 
    # do modelo de regressão. A função irá receber como argumento
    # o nome do arquivo e o delimitador (caractere que separa os
    # dados no arquivo). Como no nosso arquivo de funções nós
    # separamos os valores de caracteristicas dos valores alvo, temos
    # que atribuir essa função em 2 variaveis. Dessa maneira:
    # x: ira receber as caracteristicas
    # y: Irá receber os valores alvo
    x, y = funcoes.carregar_Dataset(nome_arquivo, delimitador)
    
    # Ira pegar o tempo atual (tempo após a execução dessa etapa
    # ) e irá subtrair com o tempo inicial com o objetivo de calcular
    # o tempo (em segundos) necessários para a execução da etapa.
    tempo_decorrido = time.time() - tempo_inicial
    
    # Print do tempo necessário para execução da etapa de carregamento do dataset
    print("Tempo de carregamento do dataset: %.2f " % tempo_decorrido, " segundos")
    
    # Ira pegar o segundo atual (inicio da etapa)
    tempo_inicial = time.time()
    
    # Ira rotular os dados categóricos da coluna 3 em formato
    # binário. 
    # x: conjuntos de valores que devem ser rotulados.
    # 3: posição da coluna que será rotulada
    x = funcoes.rotulacao(x, 3)
    
    # Ira pegar o tempo atual (tempo após a execução dessa etapa
    # ) e irá subtrair com o tempo inicial com o objetivo de calcular
    # o tempo (em segundos) necessários para a execução da etapa.
    tempo_decorrido = time.time() - tempo_inicial
    
    # Ira imprimir o tempo necessário para a execução da criação de
    # rótulos binários
    print("Tempo para criar os rótulos numéricos: %.2f "% tempo_decorrido, " segundos")
    
    # Ira pegar o segundo atual (inicio da etapa) 
    tempo_inicial = time.time()
    
    # Ira usar a função de preenchimento de dados para atribuir valores
    # a linhas nulas (NaN) usando a média aritmética como estratégia.
    # x: Conjuntos de valores que possuem dados faltantes
    # mean: estratégia do simpleimputer que irá preencher os valores faltantes
    x = funcoes.preencherDadosFaltantes(x, 'mean')
    
    # Ira pegar o tempo atual (tempo após a execução dessa etapa
    # ) e irá subtrair com o tempo inicial com o objetivo de calcular
    # o tempo (em segundos) necessários para a execução da etapa.
    tempo_decorrido = time.time() - tempo_inicial
    
    # Impressão do tempo necessário para a execução dos rótulos binários
    print("Tempo de preenchimento dos dados faltantes: %.2f"% tempo_decorrido, " segundos")
    
    # Irá pegar o segundo atual (inicio da etapa)
    tempo_inicial = time.time()
    
    # Ira utilizar a função de treino e teste nos dados do arquivo
    # escolhido para a construção do modelo. A função irá receber como
    # argumento :
    # x: conjuntos de caracteristicas (que serão atribuidos a x_train e 
    # x_test)
    # y: variável alvo (que serão atribuidos a y_train e y_test)
    # 0.2: tamanho do conjunto de testes. Nesse caso escolhemos 0.2,
    # ou seja, 20% pra testes e 80% para treino
    x_train, x_test,y_train, y_test = funcoes.treino_teste(x, y, 0.2)
    
    # Ira pegar o tempo atual (tempo após a execução dessa etapa
    # ) e irá subtrair com o tempo inicial com o objetivo de calcular
    # o tempo (em segundos) necessários para a execução da etapa.
    tempo_decorrido = time.time() - tempo_inicial
    
    # Impressão do tempo necessário para a execução de treino e teste
    print("Tempo da separação dos dados em treino e teste: %.2f "% tempo_decorrido, " segundos")
    
    # Ira pegar o segundo atual (inicio da etapa )
    tempo_inicial = time.time()
    
    # Função que irá criar a regressão com os dados do dataset
    computarRegressaoMultipla(x_train, x_test, y_train, y_test)
    
    # Ira pegar o tempo atual (tempo após a execução dessa etapa
    # ) e irá subtrair com o tempo inicial com o objetivo de calcular
    # o tempo (em segundos) necessários para a execução da etapa.
    tempo_decorrido = time.time() - tempo_inicial
    
    # Impressão do tempo necessário para a construção do modelo de regressão
    print("Tempo da criação da regressão linear multipla: %.2f"% tempo_decorrido, " segundos")



