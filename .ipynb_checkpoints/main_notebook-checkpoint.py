import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy 
import ta
import sys
import os
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0,os.path.join("C:/",os.sep,"Users","jrebe","Desktop","IA EN TRADING ALGORÍTMICO", "codigo","zigzag"))
import zigzag_pips as zg

sys.path.insert(0,os.path.join("C:/",os.sep,"Users","jrebe","Desktop","IA EN TRADING ALGORÍTMICO", "codigo"))
from generadores import generadorSemanas, generadorVentanas, completarVentana, filtroVentana, generarVentanasFiltradas
from generadores import separarVentana, diaSemana, quitarFindes, generarImagenes, prepararData

divisas = ["EUR_USD","GBP_USD"]
campo = "15min"
paths_data = [os.path.join("C:/",os.sep,"Users","jrebe","Desktop","IA EN TRADING ALGORÍTMICO",
                           "datos_forex","datos_crudos",divisa,"".join([divisa,"_",campo,".csv"])) for divisa in divisas]
n_semanas, n_vent_semana= 40, 10
len_ventana_total, len_ventana_presente = 200, 80
ymax, ymin = 0.01, -0.005
a_max, a_min = 0.015, 0.005

ventanas_presentes, ventanas_futuras = generarVentanasFiltradas(paths_data,
                                                                n_semanas = n_semanas, n_vent_semana = n_vent_semana,
                                                                len_ventana_total = len_ventana_total, len_ventana_presente = len_ventana_presente,
                                                                ymax = ymax, ymin = ymin, a_max = a_max, a_min = a_min,
                                                                price = "B.C")

