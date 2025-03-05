import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Inicializar pesos y sesgo
def inicializacion(n_features):
    pesos = np.zeros(n_features)  # Inicializar los pesos a 0
    sesgo = 0.5  # Inicializar el sesgo a 0.5
    return pesos, sesgo

# Función de activación
def sigmoide(x):
    return np.where(x >= 0, 1, 0)  # Devolver 1 si x >= 0, 0 en otro caso

# Actualizar pesos y sesgo
def nuevos_pesos(pesos, sesgo, x_i, y_real, y_predicho, tasa_aprendizaje):
    actualizacion = tasa_aprendizaje * (y_real - y_predicho) * x_i  # Calcular actualización
    pesos += (actualizacion * x_i)  # Actualizar pesos
    sesgo += actualizacion  # Actualizar sesgo
    return pesos, sesgo

# Predecir una etiqueta
def prediccion(X, pesos, sesgo):
    salida_lineal = np.dot(X, pesos) + sesgo  # w^T * x + b
    return sigmoide(salida_lineal)

# Entrenar el perceptrón
def entrenamiento(X_train, y_train, X_val, y_val, tasa_aprendizaje=0.01, n_iter=50):
    n_muestras, n_caracteristicas = X_train.shape  # Número de muestras y características
    pesos, sesgo = inicializacion(n_caracteristicas)  # Inicializar pesos y sesgo

    y_train = y_train.reset_index(drop=True)  # Reiniciar índices

    for epoca in range(n_iter):  # Iterar sobre el número de épocas
        for idx, x_i in enumerate(X_train):  # Iterar sobre las muestras
            salida_lineal = np.dot(x_i, pesos) + sesgo  # w^T * x + b
            y_predicho = sigmoide(salida_lineal)  # Aplicar la función de activación
            pesos, sesgo = nuevos_pesos(pesos, sesgo, x_i, y_train[idx], y_predicho, tasa_aprendizaje)  # Actualizar pesos y sesgo

        # Mostrar la época actual y los pesos
        print(f'Época {epoca + 1}/{n_iter} | Pesos: {pesos} | Sesgo: {sesgo}')

    return pesos, sesgo

# Cargar datasets
iris = pd.read_csv('Iris.csv')

# Preprocesar el dataset Iris (usando solo dos clases)
iris = iris[iris['Species'].isin(['Iris-setosa', 'Iris-versicolor'])]
X_iris = iris.drop(columns=['Id', 'Species'])
y_iris = iris['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

# Escalar características
escalador = StandardScaler()
X_iris = escalador.fit_transform(X_iris)

# Dividir en conjuntos de entrenamiento y prueba
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Entrenar con el dataset Iris
print("Entrenando con el dataset Iris:")
pesos_iris, sesgo_iris = entrenamiento(X_train_iris, y_train_iris, X_test_iris, y_test_iris, tasa_aprendizaje=0.01, n_iter=100)
