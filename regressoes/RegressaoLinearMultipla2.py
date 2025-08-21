# Descrição da aula: O objetivo dessa aula é tornar a passagem de caracteristicas
# mais especifica já que eliminaremos do conjunto (de caracteristicas) os valores
# que não são relevantes pro modelo de predição.

# In[1]:


# Biblioteca que possibilita a realização de cálculos matemáticos e 
# manipulação de arrays
import numpy as np

# Import do arquivo do nosso arquivo de funções
from minhasfuncoes import funcoes


#                                     Back Ward Elimination
# 
# -> Técnica que tem como objetivo eliminar variáveis independentes que não são relevantes para o modelo de regressão.
# 
# -> Imagine que você está tentando prever o preço de um carro e tem dezenas de
# informações sobre ele (ano, quilometragem, cor, número de portas, tipo de motor,
# se tem teto solar, ar condicionado, etc). Nem todas essas informações são igualmente
# importantes para determinar o preço. Algumas podem ser redundantes, e outras podem
# até confundir o modelo.
# 
# -> A Backward Elimination é uma técnica de seleção de variáveis (ou seleção de
# features). Seu objetivo é simplificar um modelo de regressão, removendo as 
# variáveis independentes (as "informações" sobre o carro) que não contribuem 
# significativamente para a capacidade de predição do modelo.
# 
# -> Interpretação: Devemos examinar os valores P das caracteristicas que indica
# a probabilidade de uma variável ser relevante para o modelo. Um valor p alto
# (maior que um limiar pré-definido, comum 0.05 ou 0.10) sugere que a variável
# é estatisticamente pouco significante e pode ser removida sem grande perda para
# o poder preditivo do modelo.

# In[ ]:


# Função que irá exibir um resumo estatistico (incluindo os valores p que
# iremos eliminar do modelo).
def computeMLRWithBackWardElimination(x, y):
    
    # É uma biblioteca python poderpsa para modelagem estatistica e econometria.
    # Enquanto o scikit-learn (onde estão LinearRegression e train_test_split)
    # é focado principalmente em Machine Learning para construir modelos preditivos
    # e algoritmos de aprendizados, o statsmodels se destaca por seu rigor estatisco
    # e detalhamento na inferência. Com ele podemos criar:
    
    # Modelos Estatísticos Abrangentes: Com statsmodels, você pode implementar:

    # Modelos de Regressão Linear: Como o OLS (Ordinary Least Squares - Mínimos        Quadrados Ordinários), que você viu no código, que é a base da Regressão Linear Múltipla.

    # Modelos Lineares Generalizados (GLM).

    # Modelos de Séries Temporais (ARIMA, GARCH, etc.).

    # Testes Estatísticos diversos.
    
    import statsmodels.api as sm
    
    # Insere uma coluna de 1s no inicio (indice 0) do array x. Isso representa
    # o termo de intercepto (constante) no modelo de regressão, que é necessário
    # para o statsmodels. OLS calcular corretamente os coeficientes e os valores
    # p. Em termos leigos: é a "base" do nosso modelo, mesmo que todas as outras
    # variáveis sejam zero. Para realizar essa etapa vamos usar o método insert
    # da biblioteca numpy. A função irá receber como parametro:
    # x: Conjunto de dados que irá receber os valores.
    # 0: A posição da coluna que os dados serão inseridos
    # 1: valores que 1s que serão inseridos no conjunto
    # axis: que indica que estamos inserindo valores em uma
    # nova coluna
    x = np.insert(x, 0, 1, axis=1)
    
    # Seleciona um subconjunto fixo de colunas do seu array x.
    # Estamos pegando a coluna 0 (a constante que acabamos de adicionar),
    # e as colunas 1, 2, 3, 4, 5, 6 do nosso array original (cada valor
    # representa uma das colunas (caractreisticas) do arquivo insurance.csv que utilizaremos
    # nesse exercicio). Observação: Esta etapa não realiza uma seleção 
    # automática baseada em relevância, nós estamos escolhendo as colunas
    # manualmente.
    XOtimo = x[:,[0, 1, 2, 3, 4]]
    
    # Cria e treina um modelo de Minimos Quadrados Ordinários (OLS) usando o
    # statsmodels. A função recebe como parametros:
    # y: é a variável que queremos prever
    # XOtimo: São as variáveis que usaremos para prever.
    # astype(float): converte os dados em float garantindo que eles estejam
    # no tipo numérico correto.
    # fit: função que irá treinar o modelo usando os valores XOtimo
    regressor = sm.OLS(y, XOtimo.astype(float)).fit()
    
    # Ira imprimir o resumo estatistico do modelo
    print(regressor.summary()) 
    
    # Irá retornar o conjunto que escolhemos ser relevantes (após avaliar
    # o p valor) 
    return XOtimo


# In[9]:


# Função que irá criar o modelo de regressão linear (modelo de
# predição). A função irá receber como argumento:
# nome_do arquivo: ira receber o nome (local) do arquivo que 
# iremos utilizar na construção do modelo.
# delimitador: Ira receber o caractere que separa os dados no
# arquivo. O argumento terá como padrão None, pois, caso o 
# separador de dados do arquivo seja a ',' iremos deixar
# esse argumento em "branco".
def regressaomultipla(nome_arquivo, delimitador=None):
    
    # Import da função LinearRegression da biblioteca sklearn.linear_model
    # que tem como objetivo construir modelos de predição de valores
    from sklearn.linear_model import LinearRegression
    
    # Carregando o dataset: Função que irá carregar o dataset. Ela irá
    # receber como parametro o nome do arquivo e o delimitador (caractere
    # que separa os dados no arquivo). 
    # Observação: Como no arquivo de funções dividimos x e y em caracteristicas
    # e alvo (transformamos eles em arrays numpy),precismaos atribuir a função em
    # 2 variáveis:
    # x: irá armazenar as caracteristicas
    # y: irá armazenar as variáveis alvo.
    x, y = funcoes.carregar_Dataset(nome_arquivo, delimitador)
    
    # Ira transformar as variáveis categóricas em rótulos binários
    # x: Conjunto de dados (array) que iremos rotular
    # 3: Posição da coluna que iremos rotular
    x = funcoes.rotulacao(x,3)
    
    # Ira preencher os dados faltantes de x (NaN)
    # x: conjunto que terá os dados faltantes preenchidos
    # 'mean' (média aritmética): estrategia que a classe SimpleImputer irá
    # utilizar para preencher os dados faltantes
    x = funcoes.preencherDadosFaltantes(x, 'mean')
    
    # Chamada da função que irá analisar e utilizar as caracteristicas
    # relevantes para o modelo. A função ira receber como parametro:
    # x: Conjunto de caracteristicas
    # y: variavel alvo que queremos prever.
    x = computeMLRWithBackWardElimination(x,y)
    
    # Função que irá separar e treinar os dados do modelo onde:
    # x_train: ira receber as caracteristicas que irão treinar o processo
    # de predição do modelo.
    # x_test: Irá receber o conjunto de caracteristicas que o modelo
    # irá usar para fazer a predição
    # y_train: Conjunto de dados reais que irá treinar as respostas do modelo
    # y_test: Dados reais da base de dados que serão comparados com a predição
    # do modelo.
    x_train, x_test, y_train, y_test = funcoes.treino_teste(x, y, 0.2)
    
    # Instância da classe (criação de objeto) de regressão linear
    regressao = LinearRegression()
    
    # Ira treinar o modelo usando as variáveis de treino
    regressao.fit(x_train, y_train)
    
    # Ira realizar a predição dos valores usando as caracteristicas
    predicao = regressao.predict(x_test)
    
    # For que irá percorrer o array de predição com o objetivo
    # de imprimir na tela os valores previstos
    for i in range(0, predicao.shape[0]):
        
        print("Predição"," Valor Real")
        
        # Predição: Dados previstos pelo modelo
        # y_test: Dados reais da base de dados.
        # round: método que irá arredondar os valores limitando
        # a quantidade de valores após a virgula.
        # 2: quantidade de valores após a virgula
        print(round(predicao[i], 2), round(y_test[i], 2))
    
    
    
    


# In[10]:


# Chamada da função que irá construir o modelo de regressão
# usando o arquivo insurance.csv
regressaomultipla('Dados/insurance.csv', ';')


#                                                  Conclusão do Back Ward Elimination 
# -> Podemos que a função definiu o x4 (region) como irrelevante para o modelo com um 
# p valor de 0.238
# 
# 
