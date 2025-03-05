import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# funciones de activacion
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def tangente(x):
    return np.tanh(x)

# inicializacion de los pesos
def inicializar_pesos(topologia):
    pesos = []
    for i in range(len(topologia) - 1):
        pesos.append(np.random.randn(topologia[i], topologia[i+1]) * 0.01)
    return pesos

# funcion de feedforward con diferentes activaciones por capa
def feedforward(X, pesos, activaciones):
    activacion_salida = X
    for i in range(len(pesos) - 1):
        z = np.dot(activacion_salida, pesos[i])
        if activaciones[i] == 'sigmoide':
            activacion_salida = sigmoide(z)
        elif activaciones[i] == 'relu':
            activacion_salida = relu(z)
        elif activaciones[i] == 'tangente':
            activacion_salida = tangente(z)

    # ultima capa con softmax
    z = np.dot(activacion_salida, pesos[-1])
    activacion_salida = softmax(z)

    return activacion_salida

# funcion de prediccion
def prediccion(X, pesos, activaciones):
    activacion_salida = feedforward(X, pesos, activaciones)
    return np.argmax(activacion_salida, axis=1)

# cargar datasets
iris = pd.read_csv('Iris.csv')

# preprocesar el dataset iris
iris = iris[iris['Species'].isin(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])]
X_iris = iris.drop(columns=['Id', 'Species'])
y_iris = pd.get_dummies(iris['Species']).values  # convertir a formato de one-hot

# escalar caracteristicas
escalador = StandardScaler()
X_iris = escalador.fit_transform(X_iris)

# dividir en conjuntos de entrenamiento y prueba
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# preguntar al usuario cuantas capas ocultas quiere
num_capas = int(input("Numero de capas ocultas: "))

# definir la topologia de la red
topologia = [X_train_iris.shape[1]]  # empezamos con el número de características de entrada

# pedir al usuario cuántas neuronas en cada capa oculta y cuál función de activación usar
activaciones = []
for i in range(num_capas):
    num_neuronas = int(input(f"Numero de neuronas en la capa {i+1}?: "))
    topologia.append(num_neuronas)
    
    print(f"Función de activación para la capa oculta {i+1}:")
    print("1. Sigmoide")
    print("2. ReLU")
    print("3. Tangente")
    opcion = input("Ingresa el número de la opción: ")
    
    if opcion == '1':
        activaciones.append('sigmoide')
    elif opcion == '2':
        activaciones.append('relu')
    elif opcion == '3':
        activaciones.append('tangente')
    else:
        print("La opción no existe, se usará ReLU en su lugar.")
        activaciones.append('relu')

# capa de salida (softmax siempre se usa aquí)
topologia.append(3)  # 3 clases para clasificación multiclase

# inicializar pesos
pesos_iris = inicializar_pesos(topologia)

# prediccion en el conjunto de prueba
y_pred = prediccion(X_test_iris, pesos_iris, activaciones)
y_test_labels = np.argmax(y_test_iris, axis=1)
precision = np.mean(y_pred == y_test_labels) * 100

# graficar la los errores y aciertos de las etiquetas 
def graficar(y_test, y_pred):
    aciertos = np.sum(y_test == y_pred)
    errores = np.sum(y_test != y_pred)

    categorias = ['Aciertos', 'Errores']
    valores = [aciertos, errores]

    plt.bar(categorias, valores, color=['green', 'red'])
    plt.title('Aciertos y Errores')
    plt.ylabel('Cantidad')
    plt.show()

print("Topologia:", topologia)
print("Funciones de activación por capa:", activaciones)
print(f"Precisión en el conjunto de prueba: {precision:.2f}%")
graficar(y_test_labels, y_pred)
