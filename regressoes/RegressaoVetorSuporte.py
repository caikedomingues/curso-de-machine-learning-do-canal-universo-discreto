

#                                 Regressão Vetor Suporte
# 
# -> Diferente da regressão linear onde desenhamos uma linha. Na regressão
# de vetor suporte, desenhamos uma "faixa", uma zona de tolerância em volta
# da linha. Essa zona tem uma espessura que a gente escolhe. A SVR então faz
# duas coisas:
# 
# -> Tenta encontrar uma linha que seja o mais plana possivel (como se fosse
# uma reta).
# 
# -> Tenta colocar a maioria dos pontos DENTRO dessa faixa de tolerância, sem
# se importar com a posição exata deles lá dentro.
# 
# -> A SVR só se preocupa com os pontos que estão fora da caixa ou que estão
# exatamente na borda da faixa. Esses pontos são os mais importantes e são chamados de vetores de suporte. Eles são os "pontos de apoio" que a SVR 
# usa para a desenhar a linha.

# Import das bibliotecas que utilizaremos em todas as funções

# In[6]:


# import dp nosso arquivo de funções (modularização) 
from minhasfuncoes import funcoes 

# Import da biblioteca que realiza cálculos matemáticos e manipulam arrays
import numpy as np


# criação da função que irá construir a regressão vetor suporte

# In[7]:


# A função que constroi regressão de vetor suporte irá receber como
# argumento:
# x: valores da caracteristica (experiência em anos)
# y: Variável alvo que queremos prever (salário)
# k: tipo do kernel (modelo da linha) que iremos utilizar
# para fazer a regressão
# d: valor que define a curvatora da linha do modelo polinomial
def computarRegressaoVetorSuporte(x, y, k, d):
    
    # Import da classe SVR da biblioteca skalearn.svm que tem como
    # objetvo criar regressões de vetor suporte
    from sklearn.svm import SVR
    
    # Instância da classe (criação do objeto) do modelo de regressão
    # de vetor de suporte
    regressor = SVR()
    
    # Antes de passarmos o parametro k, vamos verificar se o modelo
    # escolhido é o "poly" (kernel do modelo polinomiar)
    if (k == "poly"):
        
        # Se o valor de k for igual a "poly", vamos adicionar como argumento
        # no costrutor da classe o degree que define a curvatura da linha 
        # polinomial
        regressor = SVR(kernel = k, degree = d)
    
    else:
        
        # Caso seja diferente, iremos apenas passar para o k do construtor o valor atribuido a ele
        regressor = SVR(kernel = k)
    
    # x_train: Ira receber o conjunto de caracteristicas de  treino que irá ensinar o modelo a encontrar tendências e padrões entre as variáveis.
    # x_test: Ira receber o conjunto que o modelo utilizara para realizar
    # a predição
    # y_train: Dados reais da variável alvo que treinarão as respostas
    # do modelo.
    # y_test: Dados reais que serão comparados com o resultado gerado
    # pelo modelo
    # chamada da nossa função de treino e teste de dados:
    # x: conjunto de caracteristicas que serão divididas em treino e
    # e teste
    # y: variável alvo que será dividida em treino e teste 
    x_train, x_test, y_train, y_test = funcoes.treino_teste(x, y, 0.2)
    
    # Função da classe SVR que irá treinar o modelo usando os dados
    # de treino
    regressor.fit(x_train, np.ravel(y_train))
    
    # Retorno do modelo de suporte treinado 
    return regressor
    


# Criação da função que irá realizar a construção do gráfico do modelo

# In[8]:


# A função de construção de gráfico ira receber como argumento.
# xpoints: Dados reais da experiencia em anos que serão representados
# pelos pontos.
# ypoints: Dados reais dos valores que queremos prever (salário) que serão
# representados pelos pontos.
# xline:  Dados reais da experiencia em anos que serão representados
# pela linha de previsão do modelo.
# yline: Dados reais dos valores que queremos prever (salário) que serão
# representados pela linha de previsão do modelo.
def showPlot(xpoints, ypoints, xline, yline):
    
    # Import da biblioteca que permite a construção e manipulação de
    # gráficos.
    import matplotlib.pyplot as plt
   
   # função que irá criar os pontos do gráfico
   # xpoints: Valores das caracteristicas representadas por pontos.
   # ypoints: Valores da variável alvo que serão representados por
   # pontos
   # color: cor dos pontos no gráfico
    plt.scatter(xpoints, ypoints, color = 'red')
    
   # função que irá criar as linhas do gráfico
   # xline: Valores das caracteristicas que a linha ira usar para prever
   # os resultados.
   # yline: Valores da variável alvo que o modelo preveu
   # color: cor da linha no gráfico
    plt.plot(xline, yline, color='blue')
    
    # Ira definir o titulo do gráfico
    plt.title("Comparando os pontos reais com a reta produzida pela regressão vetor suporte")
    
    # Ira definir o rótulo do eixo x
    plt.xlabel("Experiência em anos")
    
    # Ira definir o rótulo do eixo y
    plt.ylabel("Salário")
    
    # Irá exibir o gráfico na tela
    plt.show()
    


# Criação do método que irá aplicar a construção da regressão vetor suporte

# In[ ]:


# A função irá receber como argumento:
# nome_arquivo: nome do arquivo (local do arquivo) que será utilizado
# na construção do modelo
# delimitador: Ira especificar o caractere que separa os dados no
# arquivo. Por padrão, esse parametro terá o None como valor, dessa
# maneira, caso o delimitador seja a ',' (delimitador padrão), não 
# será necessário especificar o delimitador.
def regressorVetorSuporte(nome_arquivo, delimitador = None):
    
    # Biblioteca que possui funções que manipulam o tempo
    import time
    
    # Ira conter o segundo atual (inicio da execucção do programa)
    tempo_inicial = time.time()
    
    # Ira chamar a nossa função de carregar dataset.
    # x: Irá receber o conjunto de caracteristicas do dataset
    # y: Ira receber a variável alvo (valores que queremos 
    # prever).
    # nome_do_arquivo: local do arquivo escolhido
    # delimitador: caractere que separa os dados no arquivo.
    x, y = funcoes.carregar_Dataset(nome_arquivo, delimitador)
    
    # Ira conter o tempo necessário para a execução do 
    # carregamento do dataset.
    tempo_necessario = time.time() - tempo_inicial
    
    # Impressão do tempo necessário para carregamento do dataset
    print("Tempo necessário para o carregamento do dataset na memória: ", round(tempo_necessario,3), " segundos")
    
    # Ira conter o segundo atual (inicio do processo de normalização)
    tempo_inicial = time.time()
    
    # Chamada da função de normalização: A maioria dos modelos de 
    # aprendizado de máquina, incluindo o SVR, funciona muito melhor
    # quando os dados estão na mesma escala. Por exemplo, se o nosso
    # x vai de 1 a 10 e o y de 40.000 a 100.000, o modelo pode ter
    # dificuldade me convergir.
    # x.reshape(-1, 1): Como vimos anteriormente, o x original é um
    # array de 1 dimensão (transformamos os dados em um array numpy 
    # no nosso arquivo de funções usando o values do pandas, com o
    # objetivo de facilitar a separação dos dados, já que dessa maneira
    # acessaremos apenas os valores). O reshape transforma em um array 2D 
    # com uma única coluna, que é o formato esperado pela maioria das
    # funções do scikit-learn.
    # funcoes.normalizacao: Chamada da função de normalização que
    # criamos em funcoes.py
    # x: Irá conter os dados de x normalizados
    # scaleX: Guarda o objeto StandardScaler que foi usado. Isso
    # é crucial por que você vai precisar dele mais tarde para reverter
    # a normalização.
    x, scaleX = funcoes.normalizacao(x.reshape(-1, 1))
    
    # Ira repetir o mesmo processo de normalização do x
    y, scaleY = funcoes.normalizacao(np.reshape(y, (-1, 1)))
    
    # Irá conter o tempo necessário para normalização
    tempo_necessario = time.time() - tempo_inicial
    
    # Impressão do tempo necessário para normalização dos dados.
    # O round irá limitar a quantidade de valores após a virgula
    print("Tempo necessário para normalização dos dados: ", round(tempo_necessario, 3), " segundos")
    
    # Ira conter o segundo atual (inicio da construção do modelo)
    tempo_inicial = time.time()
    
    # Chamada do método de construção de regressão. A função
    # ira receber como valor:
    # x: caracteristica utilizada para a predição
    # y: variável alvo que queremos prever
    # poly: kernel (modelo de regressão) que escolhemos utilizar
    # 3: nivel da curva da linha de previsão do modelo.
    svrModel = computarRegressaoVetorSuporte(x, y, "poly", 3)
    
    # Tempo necessário para a construção do modelo
    tempo_necessario = time.time() - tempo_inicial
    
    # Impressão do tempo necessário para construir o modelo
    print("Tempo necessário para construção do modelo: ",round(tempo_necessario, 3), " segundos")
    
    # Chamada da função que irá construir o gráfico de regressão.
    
    # scaleX.inverse_transform(x): Aqui, você está pegando os dados
    # de x que foram normalizados no passo anterior e usando o objeto
    # scaleX para reverter a nomalização. Isso transforma  os dados de
    # volta á sua escala original (1 a 10 anos de experiência) que é a
    # escala que faz sentido para o nosso gráfico. Este valor será o 
    # xpoints e o xline.
    
    # scaleY.inverse_transform(y): O mesmo é feito com os dados de y, desnormalizando-os de volta aos valores de salário originais. Este será o ypoints.

    # svrModel.predict(x): Função da classe SVR que irá prever os valores.
    
    # .reshape(-1, 1): Como a função inverse_transform também espera um array 2D, a gente transforma a saída do predict em uma matriz com uma única coluna.
    
    # scaleY.inverse_transform(...): A previsão do modelo (os valores de salário previsto) é desnormalizada de volta a escala de salário original,
    # para que a linha do gráfico tenha os valores de salário reais. Este será
    # o yline.
    showPlot(scaleX.inverse_transform(x), scaleY.inverse_transform(y), scaleX.inverse_transform(x), scaleY.inverse_transform(svrModel.predict(x).reshape(-1, 1)))


# In[ ]:


# Chamada da função que aplica a regressão. Usaremos o arquivo salary.csv
# para construção do modelo
regressorVetorSuporte('Dados/salary.csv', ';')

