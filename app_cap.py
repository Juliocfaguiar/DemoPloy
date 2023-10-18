# Deploy de aplicações Preditivas com Streamlit

# Imports
import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

####### Programando a Barra Superior da Aplicação WEB  #######

# Título 
st.write("*Formação Engenheiro de Machine Learning*")
st.write("*Deploy modelos de Machine Learning*")
st.write("*Deploy de Aplicações Preditivas com Streamlit*")
st.title("Regressão Logistica")

##### Programando a Barra Lateral #####

# Cabeçalho Lateral

st.sidebar.header('Dataset e Hiperparâmetros')
st.sidebar.markdown ("""**Seleciome o Dataset Desejado**""")
Dataset = st.sidebar.selectbox('Dataset',('Iris','Wine','Breast Cancer'))
Split = st.sidebar.slider('Escolha o Percentual de Divisão dos Dados em Treino e teste (Padrão = 70/30):', 0.1, 0.9, 0.70)
st.sidebar.markdown("""**Selecione os Hiperparâmetros Para o Modelo de Regressão Logística **""")
Solver = st.sidebar.selectbox('Algoritmo',('lbfgs','newton-cg','liblinear','sag'))
Penality = st.sidebar.radio("Regularização:",('nome','l1','l2','elasticnet'))
Tol = st.sidebar.text_input("Tolerância Para Critério de Parada(default = 1e-4):","1e-4")
Max_Iteration = st.sidebar.text_input("Número de Iterações (default = 50):","50")


# Dicionário Para os Hiperparâmetros
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
parameters = {'Penality':Penality,'Tol':Tol,'Max_Iteration':Max_Iteration,'Solver':Solver}

####### Função para Carregar e Preparar os Dados #######

#Função para carregar o dataset
def carrega_dataset(dataset):

    #Carrega o Dataset
    if dataset == 'Iris':
        dados = sklearn.datasets.load_iris()
    elif dataset == 'Wine':
        dados = sklearn.datasets.load_wine()
    elif dataset == 'Breast Cancer':
        dados = sklearn.datasets.load_breast_cancer()
    
    return dados

# Função para preparar os dados e fazer a divisão em treino e teste
def prepara_dados(dados,split):

    #Divide os dados de acordo com o valor de split definido pelo usuário
    x_treino, x_teste, y_treino, y_teste = train_test_split(dados.data, dados.target, test_size = float(split),random_state= 42)

    #Prepara o scaler para padronização
    scaler =  MinMaxScaler()

    #fit e transform nos dados de treino
    x_treino = scaler.fit_transform(x_treino)

    #Apenas transform nos dados de teste
    x_teste = scaler.transform(x_teste)

    return (x_treino, x_teste, y_treino,y_teste)


###### Função para o Modelo de Machine Learning #####

# Função do Modelo
def cria_modelo(parameters):

    #Extrai os dados de treino e teste
    x_treino,x_teste,y_treino,y_teste = prepara_dados(Data, Split)

    #Cria o modelo 
    clf = LogisticRegression(Penalty = parameters['Penality'],
                             solver = parameters['Solver'],
                             max_iter= int(parameters['Max_Iteration']),
                             tol= float(parameters['Tol']))
    
    #Treina o modelo
    clf = clf.fit(x_treino,y_treino)

    # Faz Previsões
    prediction = clf.predict(x_teste)

    # Calcula a acurácia
    accuracy= sklearn.metrics.accuracy_score(y_teste, prediction)

    # Calcula a confusion matrix
    cm = confusion_matrix(y_teste,prediction)

    #Dicionário como os resultados
    dict_value = {"modelo":clf,"acuracia":accuracy,"previsão":prediction,"y_real":y_teste,"Metricas":cm,"x_teste":x_teste}

    return(dict_value)

    return (x_treino, x_teste, y_treino,y_teste)

### Programa o Corpo da Aplicação Web ####

# Resumo dos dados 
st.markdown("""Resumo dos Dados""")
st.markdown("Nome do Dataset:",Dataset)

# Carrega o dataset escolhido pelo usuário
Data = carrega_dataset (Dataset)

# Extrai a variável alvo
target = Data.target_names

# Prepara o dataframe com os dados 
Dataframe = pd.DataFrame (Data.data, columns= Data.feature_names)
Dataframe['target'] = pd.Series(Data.target)
Dataframe['target labels'] = pd.Series(targets[i] for i in Data.target)

#mostra o dataset selecionado pelo usuário
st.write("Visão Geral dos Atributos")
st.write(Dataframe)