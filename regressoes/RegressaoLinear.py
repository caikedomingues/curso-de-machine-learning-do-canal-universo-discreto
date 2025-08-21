

# Descrição da aula: Nessa aula o professor irá apresentar brevemente o conceito de Regressão Linear. Ele Explicara o que é, como funciona, para que serve, como implementar usando a linguagem Python e a biblioteca Scikit-Learn e irá terminar gerando um gráfico para uma base de dados de exemplo.

# A ideia aqui é criar um modelo de predição através da regressão linear(linha 
# que mostra a relação entre variáveis numéricas) para descobrir se o modelo consegue prever o numero de visualizações de cada canal do youtube

# In[1]:


# Primeiro, vamos importar o arquivo de funções que criamos para
# facilitar o nosso processo de desenvolvimento.
from minhasfuncoes import funcoes

# Função que irá criar o modelo de regressão linear, ela recebera como
# parametro:
# x_train: Os dados de treino que irão ensinar o modelo a identificar
# os padrões utilizados em predições.
# x_test: Ira receber as caracteristicas que o modelo não visualizou
# y_train: Dados reais que também serão utilizados para treinar o modelo
# y_test: Dados reais que serão comparados com a predição do modelo.
def computarRegressaoLinear( x_train, x_test, y_train, y_test):
    
    # Import da classe LinearRegression da biblioteca sklearn.linear_model
    # que tem como objetivo criar regressões lineares com os dados da 
    # base de dados
    from sklearn.linear_model import LinearRegression
    
    # Biblioteca que possibilita a construção de inumeros tipos de gráficos
    import matplotlib.pyplot as plt
    
    # Instância da classe (criação do objeto) LinearRegression
    regressao = LinearRegression()
    
    # Treinando o modelo usando as caracteristicas e os valores reais
    regressao.fit(x_train, y_train)
    
    # Realizando a predição do modelo.
    ypredicao = regressao.predict(x_test)
    
    # Comparando os valores reais com os previsto
    print(y_test, ypredicao)
    
   # Este bloco de código tem como objetivo principal visualizar a qualidade do ajuste do seu modelo de regressão linear aos dados. Ele faz isso plotando os dados reais e a linha de previsão do modelo em um único gráfico.
    
    # Função que é usada para criar um gráfico de dispersão(scatterplot). A 
    # função irá receber como argumento:
    
    # valores do eixo x: Ira receber todos as linhas (valores)
    # da última coluna (inscritos)
    
    # valores do eixo y: ira receber os valores reais da base de dados(
    # valores das visualizações que o modelo quer prever).
    
    # color: irá definir a cor dos pontos nos gráficos
    
     # Gráfico de dispersão(scatterplot): Um gráfico de dispersão mostra
    # os pontos individuais de dados, onde a posição de cada ponto é
    # determinada pelos valores de duas variáveis. É ideal para ver
    # a distribuição de dados e possiveis relações.
    
    plt.scatter(x_test[:, -1], y_test, color='red')
    
    # Função que irá criar o gráfico de linhas que irá mostrar
    # a linha de previsão do nosso modelo.A função irá receber 
    # como parametro:
    # valores do eixo x: ira receber todas as linhas (valores) da última
    # coluna (inscritos)
    
    # Valores do eixo y: Irá receber a predição do modelo (método
    # predict com as caracteristicas de teste que são utilizadas
    # para realizar a predição)
    
    # color: Ira definir a cor das linhas.
    plt.plot(x_test[:,-1], regressao.predict(x_test), color = 'blue')
    
    # Irá definir o titulo do grafico
    plt.title('Inscritos x visualizações (SVBR)')
    
    # Ira definir o rótulo do eixo x
    plt.xlabel("Total de inscritos")
    
    # Ira definir o rótulo do eixo y
    plt.ylabel('Total de visualizações')
    
    # Ira exibir o gráfico
    plt.show()
    
# função que irá aplicar a regressão linear na base de dados
# escolhida. A função irá receber como parametro
# nome_do_arquivo: endereço da pasta que contém a base de dados
# análisada.
# delimitador = None: Ira especificar o tipo de caractere que separa
# os dados. Ele terá como valor o None, pois, não seremos obrigados
# a passar um delimitador caso o caractere separador seja a ','
def regressao_linear(nome_do_arquivo, delimitador = None):
    
    # x: Variável que irá conter as os valores de caracteristicas
    # da base de dados (como definimos no arquivo funcoes.py)
    # y: Variável que irá conter os valores alvos da base de dados
    # (como definimos no arquivo funcoes.py)
    # funcoes.carregar_Dataset: Função do arquivo de funcoes que tem
    # como objetivo carregar o dataset analisado na memória. A função
    # irá receber como parametro o nome do arquivo e o delimitador que
    # separa os dados
    x, y = funcoes.carregar_Dataset(nome_do_arquivo, delimitador)
    
    # Função que irá preencher os dados faltantes do conjunto de
    # caracteristicas. A função irá receber como argumento o
    # conjunto de valores e a estratégia que será utilizada 
    # no preenchimento dos dados (que é utilizada pelo simpleimputer
    # definido no funcoes.py)
    x = funcoes.preencherDadosFaltantes(x, 'median')
    
    # Ira transformar os rótulos categóricos do conjunto
    # x em rótulos numéricos (rótulos binários para ser mais
    # exato, já que usamos o get_dummies do pandas para aplicar
    # nos rótulos categoricos o one-hot coding)
    x = funcoes.rotulacao(x,0)
    
    
    # x_train: ira conter a parte de treino das caracteristicas
    # que ensinarão o modelo a identificar relações, tendências e
    # padrões.
    # x_test: Irá conter o conjunto de caracteristicas que o modelo
    # não visualizou. Essas caracteristicas serão utilizadas na predição
    # do modelo
    # y_train: Dados reais que serão usados para treinar o modelo (como
    # se fossem exercicios que treinai as respostas do modelo)
    # y_test: Ira conter os valores reais da base de dados que serão
    # comparados com os valores da predição.
    # funcoes.treino_teste: função que irá treinar e testar o modelo de regressão linear. A função irá receber como argumento: 
    # x: conjunto de caracteristicas
    # y: variável alvo da predição
    # 0.8: será o tamanho do conjunto de teste. Nesse caso definimos
    # 80% de teste e 20% de treinamento
    x_train, x_test, y_train, y_test = funcoes.treino_teste(x, y, 0.8)
    
    # Ira chamar a função de construção da regressão lienar utilizando
    # o conjunto de treino e teste.
    computarRegressaoLinear( x_train, x_test, y_train, y_test)

    
