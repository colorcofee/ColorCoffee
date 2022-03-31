# Models.
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Cross-Validation models.
from sklearn.model_selection import KFold

# Metrics
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Plot Matrix
import seaborn as sn
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# Tempo de execução
import time

# Esconde os warnings
import warnings

# Escrever resultados
from datetime import datetime

# Processamento em paralelo
from sklearn.utils import parallel_backend

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

RANDOM_STATE = 42
EXPORT_MATRIZ = False
DEBUG_PRINT = True
# arquivo = ''

# Define cores no print


class bcolors:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Função para configurar e retornar os classificadores


def configurarClassificadores():
    # Floresta Aleatória
    randomForest = RandomForestClassifier(n_estimators=600, max_depth=4, criterion='gini',
                                          random_state=RANDOM_STATE, class_weight="balanced_subsample")

    # DEPOIS TESTA 200 e 500
    # randomForest = RFE(randomForest, n_features_to_select=None, step=1)

    # {'activation': 'identity', 'alpha': 0.0001, 'hidden_layer_sizes': (1, 2), 'learning_rate': 'adaptive', 'random_state': 42, 'solver': 'lbfgs'}
    # Perceptron Multicamadas
    # mplc = MLPClassifier(max_iter=25000,alpha= 0.0001, activation='identity', learning_rate='adaptive',
    #                      solver='lbfgs', hidden_layer_sizes=(1, 2), random_state=RANDOM_STATE)

    mplc = MLPClassifier(max_iter=25000, alpha=0.0001, activation='identity', learning_rate='adaptive',
                         solver='lbfgs', hidden_layer_sizes=(1,2), random_state=RANDOM_STATE)
    # Naive Bayes
    mnb = MultinomialNB()

    return randomForest, mplc, mnb

# Função para configurar e aplicar o KFold, retornando predição e tempo decorrido


def aplicarKFold(classificador, x, y):
    # Configurando KFold
    kfold = KFold(n_splits=10, shuffle=False)

    # Aplicando KFold nos modelos e calculando tempo decorrido
    inicio = time.time()
    with parallel_backend('threading'):
        pred = cross_val_predict(classificador, x, y, cv=kfold)
    fim = time.time()

    tempo = np.round(fim - inicio, 4)

    return pred, tempo

# Função para mostrar em tela todas as métricas e tempo decorrido de todos os modelos


def printMetricas(pred1, tempo1, pred2, tempo2, pred3, tempo3, y, ESPACO_COR, FILTRO):
    # Pega F1_Score, Precisão e Recall.
    randomForest_f1 = f1_score(
        y, pred1, average='weighted')
    randomForest_precision = precision_score(
        y, pred1, average='weighted')
    randomForest_recall = recall_score(
        y, pred1, average='weighted')

    mplc_f1 = f1_score(
        y, pred2, average='weighted')
    mplc_precision = precision_score(
        y, pred2, average='weighted')
    mplc_recall = recall_score(
        y, pred2, average='weighted')

    mnb_f1 = f1_score(
        y, pred3, average='weighted')
    mnb_precision = precision_score(
        y, pred3, average='weighted')
    mnb_recall = recall_score(
        y, pred3, average='weighted')

    # Creates a dictionary to store Linear Models.
    randomForest = """    F1: {} %
    Precisão: {} %
    Recall": {} %
    Tempo: {} s""".format(np.round(np.mean(randomForest_f1) * 100, 1), np.round(np.mean(randomForest_precision) * 100, 1), np.round(np.mean(randomForest_recall) * 100, 1), tempo1, 4)

    mplc = """    F1: {} %
    Precisão: {} %
    Recall": {} %
    Tempo: {} s""".format(np.round(np.mean(mplc_f1) * 100, 1), np.round(np.mean(mplc_precision) * 100, 1), np.round(np.mean(mplc_recall) * 100, 1), tempo2, 4)

    nb = """    F1: {} %
    Precisão: {} %
    Recall": {} %
    Tempo: {} s""".format(np.round(np.mean(mnb_f1) * 100, 1), np.round(np.mean(mnb_precision) * 100, 1), np.round(np.mean(mnb_recall) * 100, 1), tempo3, 4)

    if(ESPACO_COR == 0):
        info = 'RGB + HSV'
    elif(ESPACO_COR == 1):
        info = 'RGB'
    elif(ESPACO_COR == 2):
        info = 'HSV'

    if(FILTRO):
        info += ' (COM FILTRO)'
    else:
        info += ' (SEM FILTRO)'

    print(f"{bcolors.BOLD}{bcolors.OKGREEN}{'--------------------'}{bcolors.ENDC}\n")
    print(f"{bcolors.BOLD}{bcolors.WARNING}{info}{bcolors.ENDC}\n")
    print(f"{bcolors.BOLD}{bcolors.BLUE}{'Random Forest'}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}{randomForest}{bcolors.ENDC}\n")
    print(f"{bcolors.BOLD}{bcolors.BLUE}{'Multi-layer Perceptron'}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}{mplc}{bcolors.ENDC}\n")
    print(f"{bcolors.BOLD}{bcolors.BLUE}{'Naive Bayes'}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}{nb}{bcolors.ENDC}\n")
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}{'--------------------'}{bcolors.ENDC}\n")

    # arquivo.write("Data Execução: {}\n".format(
    #     datetime.today().strftime('%Y-%m-%d %H:%M')))
    # arquivo.write(info)
    # arquivo.write('\n')
    # arquivo.write('Random Forest\n')
    # arquivo.write(randomForest)
    # arquivo.write('\n')
    # arquivo.write('Multi-layer Perceptron\n')
    # arquivo.write(mplc)
    # arquivo.write('\n')
    # arquivo.write('Naive Bayes\n')
    # arquivo.write(nb)
    # arquivo.write('\n')
    # arquivo.write('\n')


# Função para salvar a matriz de confusão em disco como PDF
def matrizConfusao(nome, y, y_pred, ESPACO_COR, FILTRO):
    # Calcula a Matriz de Confusão
    cm = confusion_matrix(y, y_pred)

    # Define um tamanho para a figura
    plt.figure(figsize=(10, 7))
    ax = plt.subplot()

    # Monta o dataset e seus colunas
    df_cm = pd.DataFrame(cm, index=[25, 35, 45, 55, 65, 75], columns=[
                         25, 35, 45, 55, 65, 75])

    # Configura a imagem com tamanho, fonte e cores
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, ax=ax, annot_kws={"size": 16}, cmap='Greens')

    # Define o titulo da Matriz de COnfusão
    if(ESPACO_COR == 0):
        titulo = 'RGB + HSV'
    elif(ESPACO_COR == 1):
        titulo = 'RGB'
    elif(ESPACO_COR == 2):
        titulo = 'HSV'

    # Define os rotulos e titulo
    ax.set_title(titulo)
    ax.set_xlabel('Valor Previsto')
    ax.set_ylabel('Valor Verdadeiro')

    if(ESPACO_COR == 0):
        if(FILTRO):
            caminho = 'matrizes_confusao/all/com_filtro/'
        else:
            caminho = 'matrizes_confusao/all/sem_filtro/'
    elif(ESPACO_COR == 1):
        if(FILTRO):
            caminho = 'matrizes_confusao/rgb/com_filtro/'
        else:
            caminho = 'matrizes_confusao/rgb/sem_filtro/'
    elif(ESPACO_COR == 2):
        if(FILTRO):
            caminho = 'matrizes_confusao/hsv/com_filtro/'
        else:
            caminho = 'matrizes_confusao/hsv/sem_filtro/'

    # Salva a figura
    plt.savefig("{}{}.pdf".format(caminho, nome), bbox_inches='tight')

# Função principal de Script, que irá chamar outras funções


def script(x_axis, y_axis, ESPACO_COR, FILTRO):
    # Axis
    x = x_axis
    y = y_axis

    # Configura e retorna classificadores utilizados
    randomForest, mplc, mnb = configurarClassificadores()

    # Aplica KFold e calcula tempo.
    randomForest_pred, time_random = aplicarKFold(randomForest, x, y)
    mplc_pred, time_mplc = aplicarKFold(mplc, x, y)
    mnb_pred, time_mnb = aplicarKFold(mnb, x, y)

    # Mostra as métricas em tela
    printMetricas(randomForest_pred, time_random, mplc_pred,
                  time_mplc, mnb_pred, time_mnb, y, ESPACO_COR, FILTRO)

    # Se possivel, salva as matrizes de confusão em disco
    if(EXPORT_MATRIZ):
        matrizConfusao('mnb', y, mnb_pred, ESPACO_COR, FILTRO)
        matrizConfusao('randomForest', y, randomForest_pred,
                       ESPACO_COR, FILTRO)
        matrizConfusao('mplc', y, mplc_pred, ESPACO_COR, FILTRO)
        matrizConfusao('mnb', y, mnb_pred, ESPACO_COR, FILTRO)

# Função para montar o dataset conforme necessário


def montarDataSet(FILE, ESPACO_COR):
    # Pegando a base e dividindo em treino e teste
    # , delimiter = ';')
    dataframe = pd.read_csv('./dataset/{}.csv'.format(FILE))

    # # Remove Agtron #95
    # dataframe = dataframe[dataframe.Agtron != 'Agtron 95']

    # # Remove Agtron #85
    # dataframe = dataframe[dataframe.Agtron != 'Agtron 85']

    # dataframe = dataframe[dataframe.Agtron != 'Agtron 55']

    # Deixa somente informações em RGB
    # if(ESPACO_COR == 1):
    #     dataframe = dataframe.drop(
    #         dataframe.columns[[0, 4 , 10, 11, 12, 13,  14, 15, 16, 17, 18,  19]], axis=1)
    #     # dataframe = dataframe.drop(
    #     #     dataframe.columns[[0, 4]], axis=1)

    # # Deixa somente informações em HSV
    # if(ESPACO_COR == 2):
    #     dataframe = dataframe.drop(
    #         dataframe.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], axis=1)

    if(DEBUG_PRINT):
        print(dataframe.head(2))
        print(dataframe.columns)
        print('Data possui {} colunas'.format(len(dataframe.columns)))
    

    return dataframe

# Retorna todos os rótulos


def pegarRotulos(FILE, ESPACO_COR):
    dataframe = montarDataSet(FILE, ESPACO_COR)
    array = dataframe.values

    data = array[:, 0:(len(dataframe.columns)-1)]
    labels = array[:, (len(dataframe.columns)-1)]

    if(DEBUG_PRINT):
        print("Quantidade de atributos: {}".format(len(data[0])))
        print("Labels: {}".format(set(labels)))

    return data, labels


if __name__ == '__main__':
    # Ocultando os warnings
    warnings.filterwarnings(action='ignore')
    # arquivo = open('log_classificacoes.txt'.format(
    #     datetime.today().strftime('%Y-%m-%d %H')), 'a')
    # arquivo.write('---------- NOVA CLASSIFICAÇÃO ----------\n')
    # arquivo.write('\n')
    # arquivo.write("Data Execução: {}\n\n".format(
    #     datetime.today().strftime('%Y-%m-%d %H:%M')))

    # Roda o primeiro teste SEM filtro
    FILTRO = False
    FILE = 'mega_database'
    ESPACO_COR = 0

    # while ESPACO_COR < 3:
    #     x, y = pegarRotulos(FILE, ESPACO_COR)
    #     script(x, y, ESPACO_COR, FILTRO)
    #     ESPACO_COR = ESPACO_COR + 1

    x, y = pegarRotulos(FILE, 0)
    script(x, y, 0, FILTRO)

    # # Roda o segundo teste COM filtro
    # FILTRO = True
    # FILE = 'all_comfiltro_2'
    # ESPACO_COR = 0

    # while ESPACO_COR < 3:
    #     x, y = pegarRotulos(FILE, ESPACO_COR)
    #     script(x, y, ESPACO_COR, FILTRO)
    #     ESPACO_COR = ESPACO_COR + 1

    # arquivo.write('\n---------------------------------------\n')
    # arquivo.close()
