import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import sys
import ta
from PIL import Image 

sys.path.insert(0,os.path.join("C:/",os.sep,"Users","jrebe","Desktop","IA EN TRADING ALGORÍTMICO", "codigo"))
from data_treatment import diaSemana, quitarFindes, prepararData

def generadorSemanas(data, n_sem_total = 10, shuf = False):    
    
    #Primero debemos encontrar el primer instante del primer lunes 
    k = 0
    while k>=0:
        if diaSemana(data.Fecha.iloc[k]) == "Lunes" and data.Hora.iloc[k] == "00:00":
            t0 = k
            break
        k = k + 1
    
    # Calculamos las velas que tiene una semana segun este campo temporal
    n_velas = 1
    k =  t0 + 1
    while True:
        if diaSemana(data.Fecha.iloc[k]) != "Viernes":
            if diaSemana(data.Fecha.iloc[k]) == "Lunes" and data.Hora.iloc[k] == "00:00":
                n_velas = n_velas + 1
                n_velas_totales = k - t0
                n_velas_sobrantes = k - n_velas
                break
            else: 
                n_velas = n_velas + 1
                
        if diaSemana(data.Fecha.iloc[k]) == "Viernes":
            if data.Hora.iloc[k] <= "22:00":
                n_velas = n_velas + 1
        k = k + 1

    # Ahora iteramos aleatoriamente sobre los instantes y vamos devolviendo las semanas
    k = 0   
    aux = np.arange(0,100).tolist()
    while k >= 0:
        s = random.choice(aux)
        semana = data.loc[ (data.index >= t0 + s*n_velas_totales) & (data.index < t0 + s*n_velas_totales + n_velas), ].iloc[:-1]
        semana.index = semana.index - semana.index[0]
        yield semana   
        aux.remove(s)
        k = k + 1
        
def generadorVentanas(semana, len_ventana = 30, shuf = True):
    aux = np.arange(0,len(semana)-1-len_ventana).tolist()
    while True:
        t0 = random.choice(aux)
        ventana = semana.loc[(semana.index >= t0) & (semana.index < (t0 + len_ventana)),:]
        ventana = ventana.apply(lambda col : col - col.iloc[0] if col.name in ["B.O","B.H","B.L","B.C"] else col, axis = 0)
        ventana.index = ventana.index - ventana.index[0]
        aux.remove(t0)
        yield ventana
        
def completarVentana(ventana, boll_wind = 15, wma_wind = 15):
    
    ventana_copy = ventana.copy()
    serie = ventana["B.O"]
    hband = ta.volatility.bollinger_hband(serie, window = boll_wind)
    lband = ta.volatility.bollinger_lband(serie, window = boll_wind)
    wma = ta.trend.wma_indicator(serie,window = wma_wind)
    
    ventana_copy["hband"] = hband
    ventana_copy["lband"] = lband
    ventana_copy["wma"] = wma
    
    return ventana_copy

def separarVentana(ventana, long_presente = 50):
    return ventana.iloc[:long_presente,:], ventana.iloc[long_presente:, :] 

def filtroVentana(ventana, ymax, ymin, a_max, a_min, price = "B.C"):
    
    maxprice, minprice = max(ventana.loc[:,price]),min(ventana.loc[:,price])
    maxhband, minlband = max(ventana.hband[~ventana.hband.isnull()]), min(ventana.lband[~ventana.lband.isnull()])
    
    maxval, minval = max([maxprice, maxhband]), min([minprice, minlband])
    a_serie = maxprice - minprice
        
    if maxval < ymax and minval > ymin and a_serie > a_min and a_serie < a_max:
        return True

    return False

def generarVentanasFiltradas(paths_data, n_semanas = 5, n_vent_semana = 50, 
                             len_ventana_total = 200, len_ventana_presente = 100,
                             boll_wind = 15, wma_wind = 15,
                             ymax = 0.01, ymin = -0.005,
                             a_max = 0.015, a_min = 0.005,
                             price = "B.C"):
    
    ventanas_presentes, ventanas_futuras = [], []
    print("----------------EMPEZAMOS----------------")
    for i, path in enumerate(paths_data):
        print("\n\n ****** DIVISA " + str(i+1) + "******")
        print("\nCargando y preparando datos")
        data = prepararData(path)
        print("Hecho\n")
        week_generator = generadorSemanas(data)
        for i in range(n_semanas):
            print("Semana " + str(i+1))
            semana = next(week_generator)
            window_generator = generadorVentanas(semana, len_ventana = len_ventana_total)
            for j in range(n_vent_semana):
                ventana = next(window_generator)
                ventana, ventana_futura = separarVentana(ventana, len_ventana_presente)
                ventana = completarVentana(ventana, boll_wind = boll_wind, wma_wind = wma_wind)
                if filtroVentana(ventana,ymax = ymax, ymin = ymin, a_max = a_max, a_min = a_min):
                    ventanas_presentes.append(ventana)
                    ventanas_futuras.append(ventana_futura)

    return ventanas_presentes, ventanas_futuras
    
def generar_y_guardar_Ventanas(paths_data, save_path,
                               n_semanas = 5, n_vent_semana = 50, 
                               len_ventana_total = 200, len_ventana_presente = 100,
                               boll_wind = 15, wma_wind = 15,
                               ymax = 0.01, ymin = -0.005,
                               a_max = 0.015, a_min = 0.005,
                               price = "B.C"):
    
    ventanas_presentes, ventanas_futuras = generarVentanasFiltradas(paths_data,
                                                                n_semanas = n_semanas, n_vent_semana = n_vent_semana,
                                                                len_ventana_total = len_ventana_total, len_ventana_presente = len_ventana_presente,
                                                                ymax = ymax, ymin = ymin, a_max = a_max, a_min = a_min,
                                                                price = "B.C")
    y = etiquetarVentanas(ventanas_presentes, ventanas_futuras, thr_fut = 0.003, save_path = save_path )
    generarImagenes(ventanas_presentes, ymax, ymin, directory = save_path)    

def etiquetarVentanas(ventanas_presentes, ventanas_futuras, save = True, save_path = None, return_labels = True, thr_fut = 0.005, price = "B.C"):
    y = []
    for i, ventana in enumerate(ventanas_presentes):
        ventana_f = ventanas_futuras[i]
        v = ventana.loc[:,price].iloc[-1]
        if any(ventana_f.loc[:,price] > v + 0.005): y.append(1)
        else: y.append(0)
    
    y = np.array(y)
    if save: np.save(save_path, y)
    if return_labels: return y

def generarImagenes(ventanas, ymax, ymin, directory, price = "B.C"):
    fig = plt.figure(figsize=(6, 3), dpi=96)
    for i, ventana in enumerate(ventanas):
        if i%100 == 0: print("Generando y guardando ventana " + str(i+1))
        save_path = os.path.join(directory,"".join(["im",str(i),".png"]))
        plt.plot(ventana.loc[:,price], color = "limegreen")
        plt.plot(ventana["hband"], color = "black")
        plt.plot(ventana["lband"], color = "black")
        plt.plot(ventana["wma"], color = "green")
        plt.ylim(ymin,ymax)
        plt.axis("off")
        plt.savefig(save_path,
                    bbox_inches = "tight", pad_inches = 0,
                    transparent = False, dpi = 96)
        plt.clf()
        im = Image.open(save_path)
        im = im.convert("L")
        im = im.resize((400,200))      
        im.save(save_path)
    plt.close()

def generarImagenesII(path_images = "images", divisas = ["EUR_UDS","GBP_USD"], campo = "15min", 
                      n_semanas = 10, n_vent_semana = 20, 
                      len_ventana_total = 200, len_ventana_presente = 150, 
                      pips_thr = 0.001, fut_thr = 0.0015,
                      ma_window = 15, bol_window = 15,
                      at_max = 0.005, at_min = 0, ap_max = 0.005, an_min = -0.005):

    # Preparamos outputs
    y = []
    k = 1
    for divisa in divisas:
        print(divisa)
    # Cargamos datos de la divisa
        path_data = os.path.join("C:/",os.sep,"Users","jrebe","Desktop","IA EN TRADING ALGORÍTMICO", "datos_forex","datos_crudos",divisa,"".join([divisa,"_",campo,".csv"]))
        data = prepararData(path_data)

        # Week generator
        week_generator = generadorSemanas(data)
        fig = plt.figure(figsize=(6, 3), dpi=96)
        for i in range(n_semanas):
            semana = next(week_generator)

            # Windodw generator
            window_generator = generadorVentanas(semana, len_ventana = len_ventana_total)
            for j in range(n_vent_semana): 
                ventana = next(window_generator)
                ventana, ventana_futura = separarVentana(ventana, len_ventana_presente)

                # Analizamos la ventana presente y preparamos los inputs
                serie = ventana["B.C"]
                ma = ta.trend.wma_indicator(serie, window = ma_window)
                ma = ma.loc[~ma.isna()]; ma.index = ma.index - ma.index[0]

                serie = serie.iloc[ma_window-1:]; serie.index = serie.index - serie.index[0]
                #pivotes = zg.zigzag(serie, pips_thr)
                #piv_values = zg.pivotsValues(pivotes, serie)

                hband = ta.volatility.bollinger_hband(serie, window = bol_window); 
                lband = ta.volatility.bollinger_lband(serie, window = bol_window)

                mx = max([max(hband.loc[~hband.isna()]), max(serie)])
                mn = min([min(lband.loc[~lband.isna()]), min(serie)])

                a = mx - mn

                if a > at_min and a < at_max and mx < ap_max and mn > an_min:  
                    # Mapeamos a 0 o 1 dependiendo si en el futuro ha habido algún instante en el que el precio superaba suficientemente al último precio del presente
                    v = ventana["B.C"].iloc[-1]
                    if any(ventana_futura["B.C"] > v + 0.004): y.append(1)
                    else: y.append(0)
                    
                    # Make plot and save image
                    plt.ylim(an_min,ap_max)
                    #zg.plotPivots(zg.zigzag(serie, pips_thr), serie, ax)
                    plt.plot(serie, color = "aquamarine" )
                    plt.plot(ta.trend.wma_indicator(serie,window = ma_window), color = "darkviolet" )
                    plt.plot(ta.volatility.bollinger_hband(serie, window = bol_window), color = "black")
                    plt.plot(ta.volatility.bollinger_lband(serie, window = bol_window), color = "black")   
                    plt.axis("off")
                    plt.savefig(os.path.join(path_images,"".join(["im",str(k),".png"])),
                                bbox_inches = "tight", pad_inches = 0,
                                transparent = False, dpi = 96)   
                    plt.clf()
                    
                    im = Image.open(os.path.join(path_images,"".join(["im",str(k),".png"])))
                    im = im.convert("L")
                    im = im.resize((400,200))      
                    im.save(os.path.join(path_images,"".join(["im",str(k),".png"])))
                    
                    k = k + 1

                    
        plt.close()
    y = np.array(y)
    np.save(os.path.join(path_images, "mapping.npy"), y)
    
    return y, k

def create_model():
    
    # example of a 3-block vgg style architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 400, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # example output part of the model
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[keras.metrics.TruePositives(name = "TP"), keras.metrics.TrueNegatives(name = "TN"),
                                                                         keras.metrics.FalsePositives(name = "FP"), keras.metrics.FalseNegatives(name = "FN")])
    
    return model
