
# -> Regressão Polinomial: É uma forma de de regressão linear. A diferença está
# em como ela trata as features (variáveis independentes). Ao invés
# de modelar uma relação linear direta
# entre a variável independente x e a
# variável dependente y (como na regressão linear simples: y = b0 + b1X), a regressão polinomial adiciona termos polinomiais (potências da variável independente) á equação.
# 
# -> De forma resumida a regressão polinomial tem como objetivo não 
# seguir uma linha reta, com o objetivo de ajustar de maneira mais
# flexivel a linha de previsão.
# 
# -> No final, construiremos 2 graficos de pontos com o objetivo de comparar a regressão polinomial com a regressão linear

# Criação do gráfico ira comparar os valores da predição polinomial com os dados reais da base de dados

# In[21]:


# Função que irá construir o gráfico de comparação do modelo
# polinomial. A função irá receber como argumento:
# xpontos: pontos que representam os anos de experiência (variável inde
# pendente/caracteristicas)
# ypontos: pontos que representam a variável alvo (variável dependente que
# queremos prever)
# xlinha: Linha que irá representa os anos de experiência do candidato
# ylinha: linha que representa o salario do candidato
def showplotpolinomial(xpontos, ypontos, xlinha, ylinha):
    
    # Biblioteca que permite a construção e manipulação de gráficos
    import matplotlib.pyplot as plt
    
    # Função que irá construir o gráfico de pontos: ira
    # receber como parametro o xpontos, o ypontos e o 
    # color que definirá a cor dos pontos
    plt.scatter(xpontos, ypontos, color='red')
    
    # Função que  irá criar a linha de previsão
    # do modelo. A função irá receber como parametro
    # o xlinha, o ylinha e o color que define a cor
    # da linha.
    plt.plot(xlinha, ylinha, color='blue')
    
    # Ira definir o titulo do gráfico
    plt.title("Comparando pontos reais com a reta produzida pela regressão polinomial")
    
    # ira definir o rótulo do eixo x
    plt.xlabel("Experiência em anos")
    
    # ira definir o rótulo do eixo y
    plt.ylabel('Salário')
    
    # Ira exibir o gráfico
    plt.show()


# Criação do gráfico que irá comparar os valores da predição do modelo de regressão linear com os valores reais

# In[22]:


# Função que irá construir o gráfico de comparação do modelo
# de regressão linear. A função ira receber como  parametro:
# x: experiência em anos (variável independente que queremos
# prever).
# y: variável dependente que queremos prever (salário)
# linearRegressor: Ira representar a classe de regressão
# linear que irá treinar os dados e prever os valores
def showplotlinear(X, y, linearRegressor):
    
    # Biblioteca que permite a construção e manipulação de dados
    import matplotlib.pyplot as plt

    # Função que ira construir o grafico de pontos. A função irá receber
    # como parametro o x, o y e o color que irá definir a cor dos pontos
    # no gráfico
    plt.scatter(X, y, color = 'red')
    
    # Função que irá criar a linha de previsão do modelo. A função
    # ira receber como argumento:
    # x: Os anos de experiência dos candidatos
    # A classe LinearRegression com a função predict que irá conter 
    # a caracteristica
    # color: ira definir a cor da linha
    plt.plot(X, linearRegressor.predict(X), color = 'blue') 
    
    # Irá definir o titulo do gráfico
    plt.title("Comparando pontos reais com a reta produzida pela regressão linear")
    
    # Ira definir o  rótulo do eixo x
    plt.xlabel("Experiência em anos")
    # Ira definir o r´tulo do eixo y
    plt.ylabel("Salário")
    
    # Ira exibir o gráfico
    plt.show()


# In[23]:


# Ira construir o modelo de regressão polinomial. A função irá receber
# como argumento:
# x: ira receber a variável caracteristica (variável independente)
# y: Irá receber a variável alvo (variável dependente)
# d: Valor do degree da classe PolynomialFeatures
def computarRegressaoPolinomial(x, y, d):
    
    # import da classe PolynomialFeatures da biblioteca preprocessing
    # que tem como objetivo transformar um conjunto de caracteristicas
    # existentes em um novo conjunto de caracteristicas que inclui termos
    # polinomiais e termos de interação.
    # O principal uso é permitir que modelos de regressão linear (que por sua # natureza só ajustam linhas retas) possam ajustar curvas aos seus dados. # Se a relação entre sua variável independente e a dependente não é uma linha reta, PolynomialFeatures cria as colunas necessárias para que um modelo linear consiga capturar essa curvatura.
    from sklearn.preprocessing import PolynomialFeatures
    
    # Instância da classe  PolynomialFeatures que terá como argumento
    # em seu construtor o degree que serve para controlar a complexidade
    # da curva que o seu modelo de regressão polinomial irá ajustar aos
    # dados.
    polinomio = PolynomialFeatures(degree=d)
    
    # Ira aplicar a o polinomio nos dados de caracteristicas
    xpolinomio = polinomio.fit_transform(x)
    
    # Ira importar a classe LinearRegression que cria regressões
    # lineares
    from sklearn.linear_model import LinearRegression
    
    # Instância da classe de regressão linear
    regressoalinearpolinomial = LinearRegression()
    
    # Ira treinar os dados usando as caracteristicas x com os
    # polinomios e o dados da variável alvo
    regressoalinearpolinomial.fit(xpolinomio, y)
    
    # Ira retornar o conjunto de caracteristicas polinomiais e o treinamenyo dos dados
    return xpolinomio, regressoalinearpolinomial
    

    
    


# In[ ]:


# Ira criar o modelo de regressão linear simples
# (que utiliza apenas 1 caracteristica). A função
# irá receber 2 argumentos: 
# x: caracteristica (variável independente)
# y: variável alvo (variável dependente que queremos prever)
def computarregressaolinear(x, y):
    
    # Import da classe LinearRegression
    # da biblioteca linear model que tem
    # como objetivo criar regressões lineares
    from sklearn.linear_model import LinearRegression
    
    # Import da função train_test_split da 
    # biblioteca model selection que tem
    # como objetivo separar os dados em conjuntos de treino
    # e teste.
    from sklearn.model_selection import train_test_split
    
    # Instância da classe (criação do objeto) de regressão linear 
    regressao = LinearRegression()
    
    # chamada da função que irá separar os dados em treino e teste.
    # x_train: Ira receber o conjunto de caracteristicas treino.
    # x_test: Ira receber o conjunto que será utilizado na predição do
    # modelo após o treinamento.
    # y_train: Ira receber os dados reais que irão treinar as respostas do modelo
    # y_test: Dados reais que serão comparados com a predição do modelo
    # Argumentos da função train_test_split.
    # x: ira conter os dados de caracteristica que serão divididos em treino
    # e teste
    # y: Dados da variável alvo que serão divididos em treino e teste
    # test_size: Tamanho do conjunto de teste. No nosso caso separamos
    # 20% para teste e 80% para treino
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # Função da classe LnearRegression que ira treinar o modelo
    # usando os dados de treino
    regressao.fit(x_train, y_train)
    
    # Ira retornar o modelo de predição treinado 
    return regressao  


# In[ ]:


# Irá chamar as funções necessárias para a construção dos modelos
# e dos gráficos de regressão. A função irá receber como argumento:
# nome do arquivo: Irá receber o local do arquivo que estamos utilizando
# na construção do modelo
# delimitador: Irá receber o caractere que separa os dados no arquivo.
# esse argumento terá o None como valor padrão, pois, caso o delimitador
# seja o padrão (',') não precisaremos passar o delimitador como argumento.
def progressaopolinomial(nome_arquivo, delimitador = None):
    
    # Irá conter funções que permitem a manipulação do tempo
    import time
    
    # Import do nosso arquivo de funções (modularização)
    from minhasfuncoes import funcoes
    
    # Ira pegar o segundo que a execução começa
    tempo_inicial = time.time()
    
    # Ira chamar a função que carrega datasets na memória. A função irá receber como argumento
    # o nome do arquivo e o seu delimitador.
    # Como no nosso arquivo de funções transformamos o x e o y em arrays numpy 
    # (através da função values do pandas),
    # precisamos agora atribuir a função em
    # 2 variáveis: 
    # x: ira receber o conjunto de caracteristicas
    # y: Irá receber o alvo da predição.
    x, y = funcoes.carregar_Dataset(nome_arquivo, delimitador)
    
    # Ira calcular o tempo necessário para a execução subtraindo o segundo atual (no
    # final do processo) com o tempo inicial
    # (segundo que iniciou o processo)
    tempo_percorrido = time.time() - tempo_inicial
    
    # Impressão do tempo necessário para executar
    # o carregamento do dataset na memória
    print("Tempo necessário para o carregamento do dataset %.2f"% tempo_percorrido, " segundos")
    
    # Ira pegar o segundo atual (inicio
    # do processo de construção da regressão
    # linear)
    tempo_inicial = time.time()
    
    # Irá chamar a função que constroi a regressão
    # linear. A função irá receber como argumento:
    # x: Caracteristicas (variável independente)
    # y: Variável alvo (variável indepente)
    regressaoLinear = computarregressaolinear(x, y)
    
    # Ira calcular o tempo necessário para a execução subtraindo o segundo atual (no
    # final do processo) com o tempo inicial
    # (segundo que iniciou o processo)
    tempo_percorrido = time.time() - tempo_inicial
    
    # Impressão do tempo necessário a execução da construção de previsão do modelo.
    print("Tempo necessário para a construção da regressão linear: %.2f"% tempo_percorrido, " segundos")
    
    # Ira pegar o segundo atual (inicio do processo
    # de construção da regressão polinomial)
    tempo_inicial = time.time()
    
    # Ira chamar a função que constroi regressões polinomiais.
    # A função irá receber como argumento:
    # x: Caracteristicas (variável independente)
    # y: variável alvo (variável dependente)
    # 4: Nivel de complexidade da curva de predição
    # que irá percorrer os dados
    xpoly, polylinear = computarRegressaoPolinomial(x, y, 4)
    
    # Ira calcular o tempo necessário para a execução subtraindo o segundo atual (no
    # final do processo) com o tempo inicial
    # (segundo que iniciou o processo)
    tempo_percorrido = time.time() - tempo_inicial
    
    # Impressão do tempo necessário para construção de previsão do modelo
    print("Tempo necessário para a construção do modelo polinomial: %.2f"% tempo_percorrido, " segundos")
    
    # Chamada da função que irá construir o gráfico da regressão
    # linear. A função ira receber como argumento:
    # x: Caracteristica
    # y: variável alvo
    # regressaoLinear: Classe que irá conter os dados
    # de treino e teste para predição do modelo
    showplotlinear(x, y,regressaoLinear)
    
    # Ira construir o gráfico de regressão polinomial. A função ira receber como
    # argumento:
    # x: caracteristicas
    # y: Variável alvo
    # polynear: Classe que realiza a construção do modelo de 
    # regressão polinomial com função predict que terá o 
    # xpoly (caracteristicas de teste) como argumento utilizado
    # na predição.
    showplotpolinomial(x, y,x, polylinear.predict(xpoly))


# In[ ]:


# Chamada da função de construção das regressões.
# A função irá receber o nome do arquivo e o 
# delimitador
progressaopolinomial('Dados/salary.csv', ";")


#                                             Conclusão dos gráficos
#                                             
#                                             Interpretação do gráfico
# pontos vermelhos: Valores reais da base de dados
# 
# linha de predição: representa os valores que o modelo conseguiu prever.
# 
#                                             Regressão Linear
# -> O gráfico de regressão linear mostra uma tendência no aumentio de salário
# de funcionários com mais experiência.
# 
# -> Podemos observar que a regressão linear conseguiu prver alguns valores
# (basta observar os pontos presentes na reta) entre os valores 4 e 10.
# 
# -> Há uma pequena parcela de pontos(valores reais) que a reta não conseguiu chegar nem perto do valor de previsão.
# Porém, a regressão linear chegou próximo da maioria dos valores reais presentes no gráfico.
# 
# -> O modelo teve dificuldade de prever os valores da faixa entre o 4 e o 6 
# 
# 
#                                             Regressão Polinomial
# 
# -> Mostra uma tendência no aumento de salário de funcionários que possuem mais experiência
# 
# -> conseguiu prever uma pequena parcela de valores, porém também chegou perto de alguns valores
# (igual a regressão linear)
# 
# -> Também teve dificuldades em prever valores na faixa entre o 4 e o 6.
# 
# 
#                                              Comparação entre os modelo
# 
# -> Para mim, ambos o modelo alcançaram praticamente o mesmo resultado, pois
# ambos preveram com exatidão uma pequena parcela (arrisco a dizer que preveram
# a mesma quantidade de valores) e chegaram próximo de prever as demais, o que é
# ótimo para o modelo, já que é muito dificil prever com exatidão os valores, logo,
# devemos avaliar o quão próximo o modelo chegou de acertar o valor. Porém, como
# a regressão polinomial procura se "ajustar" melhor aos dados, talvez ele tenha
# conseguido alcançar mais valores (algo que visualmente eu não consigo ver com muita clareza)
# 
# -> Ambos modelos tiveram dificuldade em prever valores na faixa entre o 4 e o 6
