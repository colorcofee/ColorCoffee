from matplotlib import image
import statistics
import cv2
import numpy as np
import pandas as pd
import os
import csv
from sklearn.cluster import MiniBatchKMeans

# Esconde os warnings
import warnings

# Parametros
DIVISOR = 4
RANDOM_STATE = 42
DEBUG_PRINT = False

HEADER = ['media_RGB_cafe', 'media_R_cafe', 'media_G_cafe', 'media_B_cafe',
          'mediana_RGB_cafe', 'mediana_R_cafe', 'mediana_G_cafe', 'mediana_B_cafe',
          'moda_RGB_cafe', 'moda_R_cafe', 'moda_G_cafe', 'moda_B_cafe',
          'desvio_RGB_cafe', 'desvio_R_cafe', 'desvio_G_cafe', 'desvio_B_cafe',
          'media_HSV_cafe', 'media_H_cafe', 'media_S_cafe', 'media_V_cafe',
          'mediana_HSV_cafe', 'mediana_H_cafe', 'mediana_S_cafe', 'mediana_V_cafe',
          'moda_HSV_cafe', 'moda_H_cafe', 'moda_S_cafe', 'moda_V_cafe',
          'desvio_HSV_cafe', 'desvio_H_cafe', 'desvio_S_cafe', 'desvio_V_cafe',
          'media_LAB_cafe', 'media_L_cafe', 'media_A_cafe', 'media_B_cafe',
          'mediana_LAB_cafe', 'mediana_L_cafe', 'mediana_A_cafe', 'mediana_B_cafe',
          'moda_LAB_cafe', 'moda_L_cafe', 'moda_A_cafe', 'moda_B_cafe',
          'desvio_LAB_cafe', 'desvio_L_cafe', 'desvio_A_cafe', 'desvio_B_cafe',
          'media_cafe_cinza', 'mediana_cafe_cinza', 'moda_cafe_cinza', 'desvio_cafe_cinza',

          'media_RGB_folha', 'media_R_folha', 'media_G_folha', 'media_B_folha',
          'mediana_RGB_folha', 'mediana_R_folha', 'mediana_G_folha', 'mediana_B_folha',
          'moda_RGB_folha', 'moda_R_folha', 'moda_G_folha', 'moda_B_folha',
          'desvio_RGB_folha', 'desvio_R_folha', 'desvio_G_folha', 'desvio_B_folha',
          'media_HSV_folha', 'media_H_folha', 'media_S_folha', 'media_V_folha',
          'mediana_HSV_folha', 'mediana_H_folha', 'mediana_S_folha', 'mediana_V_folha',
          'moda_HSV_folha', 'moda_H_folha', 'moda_S_folha', 'moda_V_folha',
          'desvio_HSV_folha', 'desvio_H_folha', 'desvio_S_folha', 'desvio_V_folha',
          'media_LAB_folha', 'media_L_folha', 'media_A_folha', 'media_B_folha',
          'mediana_LAB_folha', 'mediana_L_folha', 'mediana_A_folha', 'mediana_B_folha',
          'moda_LAB_folha', 'moda_L_folha', 'moda_A_folha', 'moda_B_folha',
          'desvio_LAB_folha', 'desvio_L_folha', 'desvio_A_folha', 'desvio_B_folha',
          'media_folha_cinza', 'mediana_folha_cinza', 'moda_folha_cinza', 'desvio_folha_cinza',

          # 'media_1_4_RGB_cafe', 'media_2_4_RGB_cafe', 'media_3_4_RGB_cafe', 'media_4_4_RGB_cafe',
          # 'media_1_4_R_cafe', 'media_2_4_R_cafe', 'media_3_4_R_cafe', 'media_4_4_R_cafe',
          # 'media_1_4_G_cafe', 'media_2_4_G_cafe', 'media_3_4_G_cafe', 'media_4_4_G_cafe',
          # 'media_1_4_B_cafe', 'media_2_4_B_cafe', 'media_3_4_B_cafe', 'media_4_4_B_cafe',
          # 'media_1_4_C_cafe', 'media_2_4_C_cafe', 'media_3_4_C_cafe', 'media_4_4_C_cafe',
          ]
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

# Função para extrair RGB


def extrairRGB(img):
    # Divide e extrai os canais da imagem
    R, G, B = cv2.split(img)

    return (R, G, B)

# Função para extrair HSV


def extrairHSV(rgb_img):
    # Converte imagem para HSV
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

    # Divide e extrai os canais da imagem
    H, S, V = cv2.split(hsv_img)

    return (H, S, V)

# Função para extrair LAB


def extrairLAB(rgb_img):
    # Converte imagem para LAB
    LAB_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)

    # Divide e extrai os canais da imagem
    L, A, B = cv2.split(LAB_img)

    return (L, A, B)


def mode(x):
    vals, counts = np.unique(x, return_counts=True)

    # find mode
    mode_value = np.argwhere(counts == np.max(counts))

    # print list of modes
    print(vals[mode_value].flatten().tolist()[0])
    return vals[mode_value].flatten().tolist()[0]


def getValores(v):
    media_folha_RGB = np.mean(v)
    mediana_folha_RGB = np.median(v)
    moda_folha_RGB = mode(v)
    desvio_folha_RGB = np.std(v)

    return (media_folha_RGB, mediana_folha_RGB, moda_folha_RGB, desvio_folha_RGB)


def chunks(lista, n):
    inicio = 0
    for i in range(n):
        final = inicio + len(lista[i::n])
        yield lista[inicio:final]
        inicio = final


def criarDataSet(filtros):
    data = []
    dataframe = pd.read_csv('./lista_imagens.csv', delimiter=';')

    for imagem in dataframe.values:
        if(imagem[5] == True and imagem[4] != 'Lumix' and imagem[4] != 'Lumix' and imagem[4] != 'Motorola X4' and imagem[2] != 95 and imagem[2] != 85):

            img_cafe = cv2.imread('./data_fotos/{}'.format(imagem[0]))
            img_folha = cv2.imread('./data_fotos/{}'.format(imagem[1]))

            valor_agtron = imagem[2]

            # Extraindo todas informações da folha

            R_f, G_f, B_f = extrairRGB(img_folha)
            H_f, S_f, V_f = extrairHSV(img_folha)
            L_f, A_f, LB_f = extrairLAB(img_folha)
            folha_cinza = extrairCinza(img_folha)

            media_folha_RGB, mediana_folha_RGB, moda_folha_RGB, desvio_folha_RGB = getValores([
                                                                                              R_f, G_f, B_f])
            media_folha_HSV, mediana_folha_HSV, moda_folha_HSV, desvio_folha_HSV = getValores([
                                                                                              H_f, S_f, V_f])
            media_folha_LAB, mediana_folha_LAB, moda_folha_LAB, desvio_folha_LAB = getValores([
                                                                                              L_f, A_f, LB_f])
            media_folha_cinza, mediana_folha_cinza, moda_folha_cinza, desvio_folha_cinza = getValores(
                folha_cinza)

            # Extraindo todas informações do café

            R_c, G_c, B_c = extrairRGB(img_cafe)
            H_c, S_c, V_c = extrairHSV(img_cafe)
            L_c, A_c, LB_c = extrairLAB(img_cafe)
            cafe_cinza = extrairCinza(img_cafe)

            media_cafe_RGB, mediana_cafe_RGB, moda_cafe_RGB, desvio_cafe_RGB = getValores([
                                                                                          R_c, G_c, B_c])
            media_cafe_HSV, mediana_cafe_HSV, moda_cafe_HSV, desvio_cafe_HSV = getValores([
                                                                                          H_c, S_c, V_c])
            media_cafe_LAB, mediana_cafe_LAB, moda_cafe_LAB, desvio_cafe_LAB = getValores([
                                                                                          L_c, A_c, LB_c])
            media_cafe_cinza, mediana_cafe_cinza, moda_cafe_cinza, desvio_cafe_cinza = getValores(
                cafe_cinza)

            # # Monta uma linsta com as informações extraídas
            geral = [media_cafe_RGB, np.mean(R_c), np.mean(G_c), np.mean(B_c),
                     mediana_cafe_RGB, np.median(
                         R_c), np.median(G_c), np.median(B_c),
                     moda_cafe_RGB, mode(R_c), mode(G_c), mode(B_c),
                     desvio_cafe_RGB, np.std(R_c), np.std(G_c), np.std(B_c),
                     media_cafe_HSV, np.mean(H_c), np.mean(S_c), np.mean(V_c),
                     mediana_cafe_HSV, np.median(
                         H_c), np.median(S_c), np.median(V_c),
                     moda_cafe_HSV, mode(H_c), mode(S_c), mode(V_c),
                     desvio_cafe_HSV, np.std(H_c), np.std(S_c), np.std(V_c),
                     media_cafe_LAB, np.mean(L_c), np.mean(A_c), np.mean(LB_c),
                     mediana_cafe_LAB, np.median(
                         L_c), np.median(A_c), np.median(LB_c),
                     moda_cafe_LAB, mode(L_c), mode(A_c), mode(LB_c),
                     desvio_cafe_LAB, np.std(L_c), np.std(A_c), np.std(LB_c),
                     media_cafe_cinza, mediana_cafe_cinza, moda_cafe_cinza, desvio_cafe_cinza,

                     media_folha_RGB, np.mean(R_f), np.mean(G_f), np.mean(B_f),
                     mediana_folha_RGB, np.median(
                         R_f), np.median(G_f), np.median(B_f),
                     moda_folha_RGB, mode(R_f), mode(G_f), mode(B_f),
                     desvio_folha_RGB, np.std(R_f), np.std(G_f), np.std(B_f),
                     media_folha_HSV, np.mean(H_f), np.mean(S_f), np.mean(V_f),
                     mediana_folha_HSV, np.median(
                         H_f), np.median(S_f), np.median(V_f),
                     moda_folha_HSV, mode(H_f), mode(S_f), mode(V_f),
                     desvio_folha_HSV, np.std(H_f), np.std(S_f), np.std(V_f),
                     media_folha_LAB, np.mean(
                         L_f), np.mean(A_f), np.mean(LB_f),
                     mediana_folha_LAB, np.median(
                         L_f), np.median(A_f), np.median(LB_f),
                     moda_folha_LAB, mode(L_f), mode(A_f), mode(LB_f),
                     desvio_folha_LAB, np.std(L_f), np.std(A_f), np.std(LB_f),
                     media_folha_cinza, mediana_folha_cinza, moda_folha_cinza, desvio_folha_cinza
                     ]

            r_div_c = list(
                chunks(cv2.calcHist([R_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            g_div_c = list(
                chunks(cv2.calcHist([G_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            b_div_c = list(
                chunks(cv2.calcHist([B_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))

            h_div_c = list(
                chunks(cv2.calcHist([H_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            s_div_c = list(
                chunks(cv2.calcHist([S_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            v_div_c = list(
                chunks(cv2.calcHist([V_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))

            l_div_c = list(
                chunks(cv2.calcHist([L_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            a_div_c = list(
                chunks(cv2.calcHist([A_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            lb_div_c = list(
                chunks(cv2.calcHist([LB_c], [0], None, [256], [0, 256]).ravel(), DIVISOR))

            c_div_c = list(chunks(cv2.calcHist([cafe_cinza[0]], [
                           0], None, [256], [0, 256]).ravel(), DIVISOR))

            # DIVISÃO RGB
            for i in range(DIVISOR):
                if(not str(i) + '_R_cafe' in HEADER):
                    HEADER.append(str(i) + '_R_cafe')
                geral.append(np.mean(r_div_c[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_G_cafe' in HEADER):
                    HEADER.append(str(i) + '_G_cafe')
                geral.append(np.mean(g_div_c[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_B_cafe' in HEADER):
                    HEADER.append(str(i) + '_B_cafe')
                geral.append(np.mean(b_div_c[i]))

            # DIVISÃO HSV
            for i in range(DIVISOR):
                if(not str(i) + '_H_cafe' in HEADER):
                    HEADER.append(str(i) + '_H_cafe')
                geral.append(np.mean(h_div_c[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_S_cafe' in HEADER):
                    HEADER.append(str(i) + '_S_cafe')
                geral.append(np.mean(s_div_c[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_V_cafe' in HEADER):
                    HEADER.append(str(i) + '_V_cafe')
                geral.append(np.mean(v_div_c[i]))

            # DIVISÃO LAB
            for i in range(DIVISOR):
                if(not str(i) + '_L_cafe' in HEADER):
                    HEADER.append(str(i) + '_L_cafe')
                geral.append(np.mean(l_div_c[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_A_cafe' in HEADER):
                    HEADER.append(str(i) + '_A_cafe')
                geral.append(np.mean(a_div_c[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_LB_cafe' in HEADER):
                    HEADER.append(str(i) + '_LB_cafe')
                geral.append(np.mean(lb_div_c[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_cinza_cafe' in HEADER):
                    HEADER.append(str(i) + '_cinza_cafe')
                geral.append(np.mean(c_div_c[i]))

            r_div_f = list(
                chunks(cv2.calcHist([R_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            g_div_f = list(
                chunks(cv2.calcHist([G_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            b_div_f = list(
                chunks(cv2.calcHist([B_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))

            h_div_f = list(
                chunks(cv2.calcHist([H_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            s_div_f = list(
                chunks(cv2.calcHist([S_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            v_div_f = list(
                chunks(cv2.calcHist([V_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))

            l_div_f = list(
                chunks(cv2.calcHist([L_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            a_div_f = list(
                chunks(cv2.calcHist([A_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))
            lb_div_f = list(
                chunks(cv2.calcHist([LB_f], [0], None, [256], [0, 256]).ravel(), DIVISOR))

            c_div_f = list(chunks(cv2.calcHist([folha_cinza[0]], [
                           0], None, [256], [0, 256]).ravel(), DIVISOR))

            # DIVISÃO RGB
            for i in range(DIVISOR):
                if(not str(i) + '_R_folha' in HEADER):
                    HEADER.append(str(i) + '_R_folha')
                geral.append(np.mean(r_div_f[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_G_folha' in HEADER):
                    HEADER.append(str(i) + '_G_folha')
                geral.append(np.mean(g_div_f[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_B_folha' in HEADER):
                    HEADER.append(str(i) + '_B_folha')
                geral.append(np.mean(b_div_f[i]))

            # DIVISÃO HSV
            for i in range(DIVISOR):
                if(not str(i) + '_H_folha' in HEADER):
                    HEADER.append(str(i) + '_H_folha')
                geral.append(np.mean(h_div_f[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_S_folha' in HEADER):
                    HEADER.append(str(i) + '_S_folha')
                geral.append(np.mean(s_div_f[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_V_folha' in HEADER):
                    HEADER.append(str(i) + '_V_folha')
                geral.append(np.mean(v_div_f[i]))

            # DIVISÃO LAB
            for i in range(DIVISOR):
                if(not str(i) + '_L_folha' in HEADER):
                    HEADER.append(str(i) + '_L_folha')
                geral.append(np.mean(l_div_f[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_A_folha' in HEADER):
                    HEADER.append(str(i) + '_A_folha')
                geral.append(np.mean(a_div_f[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_LB_folha' in HEADER):
                    HEADER.append(str(i) + '_LB_folha')
                geral.append(np.mean(lb_div_f[i]))

            for i in range(DIVISOR):
                if(not str(i) + '_cinza_folha' in HEADER):
                    HEADER.append(str(i) + '_cinza_folha')
                geral.append(np.mean(c_div_f[i]))

            # Arrendonda os valores da lista, para que tenham somente 3 casa decimais
            geral_arredondado = [round(num, 3) for num in geral]

            if(not 'Flash' in HEADER):
                HEADER.append('Flash')
            if( 'flash' in imagem[3]):
                geral_arredondado.append(0)
            else:
                geral_arredondado.append(1)

            # Adiciona a lista o valor Agtron da amostra, que está no nome do arquivo
            if(not 'Agtron' in HEADER):
                HEADER.append('Agtron')
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
    FILE = 'mega_database'
    header, data = criarDataSet(FILTRO)
    exportarCSV(header, data, FILE)
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}{'--------------------'}{bcolors.ENDC}\n")
