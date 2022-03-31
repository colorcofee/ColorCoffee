from matplotlib import image
import cv2
import numpy as np
import pandas as pd
import os
import csv
from sklearn.cluster import MiniBatchKMeans

# Esconde os warnings
import warnings

# Parametros
RANDOM_STATE = 42
DEBUG_PRINT = False
# HEADER = ['RGB_medio_cafe', 'R_medio_cafe', 'G_medio_cafe', 'B_medio_cafe', 'RGB_std_cafe', 'R_std_cafe', 'G_std_cafe', 'B_std_cafe',
#                             'RGB_medio_folha', 'RGB_std_folha', 'HSV_medio_cafe', 'H_medio_cafe', 'S_medio_cafe', 'V_medio_cafe', 'HSV_std_cafe', 'H_std_cafe', 'S_std_cafe', 'V_std_cafe',
#                             'HSV_medio_folha', 'HSV_std_folha', 'Agtron']

HEADER = ['media_cinza_cafe', 'R_medio_cafe', 'G_medio_cafe', 'B_medio_cafe',  'H_medio_cafe', 'S_medio_cafe', 'V_medio_cafe',
                            'media_cinza_folha',  'Agtron']
# Define cores no print
class bcolors:
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def extrairCinza(img):
    ay = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c = cv2.split(ay)
    return c

# Função para extrair RGB e/ou Equalizar Histograma
def extrairRGB(img, equalizar=False):

    if(equalizar):
        # Convertendo para HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Dividindo a imagem HSV em diferentes canais
        h, s, v = cv2.split(hsv_img)

        # Aplicando equalização de histograma no canal V
        v_equalized = cv2.equalizeHist(v)

        # Unificando os canais H, S e V com equalização aplicada
        hsv_img_equalized = cv2.merge((h, s, v_equalized))

        # Convertendo imagem HSV equalizada em RGB
        img = cv2.cvtColor(hsv_img_equalized, cv2.COLOR_HSV2BGR)

    # Divide e extrai os canais da imagem
    R, G, B = cv2.split(img)

    return (R, G, B)

# Função para extrair HSV e/ou Equalizar Histograma
def extrairHSV(rgb_img, equalizar=False):
    # Converte imagem para HSV
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

    # Divide e extrai os canais da imagem
    H, S, V = cv2.split(hsv_img)

    # Equaliza somente o canal V e retorna
    if(equalizar):
        v_equalized = cv2.equalizeHist(V)
        return (H, S, v_equalized)
    else:
        return (H, S, V)

# Funçao para aplicar quantização de cores
def quantizacaoCores(rgb_img, n_clusters=6):
    # Pega tamanho da imagem
    (h, w) = rgb_img.shape[:2]

    # converte a imagem do espaço de cores RGB para o espaço de cores L*a*b*
    # -- já que estaremos agrupando usando k-means # que é baseado na distância euclidiana, usaremos o
    # L*a* b* espaço de cor onde a distância euclidiana implica
    # significado perceptivo
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)

    # remodela a imagem em um vetor para que o k-means possa ser aplicado
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # Aplica o KMeans
    clt = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = clt.fit_predict(img)

    # A imagem reduzida a cores
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # remodela o vetor para imagem novamente
    quant = quant.reshape((h, w, 3))

    # converte de L*a*b* para RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    # Retorna imagem com cores reduzidas
    return quant

# Função para criar DataSet aplicando filtros
# Essa função percorre a pasta de disposivos -> condições de iluminação -> e tipo (café ou folha)
# As imagens que foram tiradas em uma mesma condição de iluminação, possuem os mesmos valores para folha
def criarDataSet(filtros):
    data = []
    dataframe = pd.read_csv('./lista_imagens.csv', delimiter = ';')
    
    for imagem in dataframe.values:
        if(imagem[5] == True and imagem[4] != 'Lumix' and imagem[4] != 'Lumix' and imagem[4] != 'Motorola X4'):       
            img_cafe = cv2.imread('./data_fotos/{}'.format(imagem[0]))
            img_folha = cv2.imread('./data_fotos/{}'.format(imagem[1]))
            valor_agtron = imagem[2]

            if(filtros):
                # Aplica suaviação pela mediana
                img_cafe = cv2.medianBlur(src=img_cafe, ksize=9)
                img_folha = cv2.medianBlur(src=img_folha, ksize=9)

                # Aplica o processo de quantização de cores
                img_cafe = quantizacaoCores(img_cafe, n_clusters=6)
                img_folha = quantizacaoCores(img_folha, n_clusters=6)

            # Opção folha é SEMPRE extraída primeiro.
            # Se a opção for do tipo "FOLHA", que determina a condição de iluminação
            # Extrair os canais RGB e HSV, calculando suas respectivas médias e desvio padrão
            
            R_f, G_f, B_f = extrairRGB(img_folha, equalizar=filtros)
            media_folha_RGB = np.mean([R_f, G_f, B_f])
            desvio_folha_RGB = np.std([R_f, G_f, B_f])

            H_f, S_f, V_f = extrairHSV(img_folha, equalizar=filtros)
            media_folha_HSV = np.mean([H_f, S_f, V_f])
            desvio_folha_HSV = np.std([H_f, S_f, V_f])

            # Se for do tipo café
            # Extrair os canais RGB e HSV, calculando suas respectivas médias e desvio padrão
            
            R_c, G_c, B_c = extrairRGB(img_cafe, equalizar=filtros)
            H_c, S_c, V_c = extrairHSV(img_cafe, equalizar=filtros)

            media_cafe_RGB = np.mean([R_c, G_c, B_c])
            desvio_cafe_RGB = np.std([R_c, G_c, B_c])

            media_cafe_HSV = np.mean([H_c, S_c, V_c])
            desvio_cafe_HSV = np.std([H_c, S_c, V_c])

            cinza_cafe = extrairCinza(img_cafe)
            cinza_folha = extrairCinza(img_folha) 
            
            print(cinza_cafe)

            # Monta uma linsta com as informações extraídas
            geral = [np.mean(cinza_cafe), np.mean(R_c), np.mean(G_c), np.mean(B_c),np.mean(H_c), np.mean(S_c), np.mean(V_c), np.mean(cinza_folha)]

            # Arrendonda os valores da lista, para que tenham somente 3 casa decimais
            geral_arredondado = [round(num, 3) for num in geral]

            # Adiciona a lista o valor Agtron da amostra, que está no nome do arquivo
            geral_arredondado.append('Agtron {}'.format(valor_agtron))

            # Adiciona essa linha a lista de dados
            data.append(geral_arredondado)
    return HEADER,  data

# Função para exportar arquivo CSV
def exportarCSV(header, data, name):
    with open('./dataset/{}.csv'.format(name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # Escreve o cabeçalho (header)
        writer.writerow(header)
        # Escreve todas as linhas (data)
        writer.writerows(data)


if __name__ == '__main__':
    # Ocultando os warnings
    # warnings.filterwarnings(action='ignore')
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}{'--------------------'}{bcolors.ENDC}\n")
    print(f"{bcolors.BOLD}{bcolors.WARNING}{'Criando dataset SEM filtro'}{bcolors.ENDC}\n")

    # Roda o primeiro teste SEM filtro
    FILTRO = False
    FILE = 'all_semfiltro_2'
    header, data = criarDataSet(FILTRO)
    exportarCSV(header, data, FILE)
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}{'--------------------'}{bcolors.ENDC}\n")

    # print(f"{bcolors.BOLD}{bcolors.OKGREEN}{'--------------------'}{bcolors.ENDC}\n")
    # print(f"{bcolors.BOLD}{bcolors.WARNING}{'Criando dataset COM filtro'}{bcolors.ENDC}\n")

    # # Roda o segundo teste COM filtro
    # FILTRO = True
    # FILE = 'all_comfiltro_2'
    # header, data = criarDataSet(FILTRO)
    # exportarCSV(header, data, FILE)
    # print(f"{bcolors.BOLD}{bcolors.OKGREEN}{'--------------------'}{bcolors.ENDC}\n")
