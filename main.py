
# Importo librerías a utilizar

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential # Nos permite crear la red
from keras.layers import Dense      # Nos deja personalizar capas de la red


# Cargamos el dataset

DataSet = pd.read_excel('/home/vale/PycharmProjects/Fire_soundTP/Fire_Dataset.xlsx')
print(DataSet.head())  # Reviso que se haya cargado

print("Dataset info")
DataSet.info()  # Veo si hay valores null y el tipo de variables

print("Valores NaN")
print(DataSet.isna().sum())  # Veo si hay valores NaN

print("Estadísticas")
print(DataSet.describe())  # Obtengo estadísticas de los datos

oe = OrdinalEncoder()
DataSet['FUEL'] = oe.fit_transform(DataSet[['FUEL']])  # Ordinal encoden me convierte de categorica a array de enteros

# Con ésto puedo ver la interelación entre las variables:

plt.figure(figsize=(10, 5))
sns.heatmap(DataSet.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.show()

# Defino los vectores

feature_cols = ['SIZE', 'FUEL', 'DISTANCE', 'DESIBEL', 'AIRFLOW', 'FREQUENCY', 'FREQUENCY']
X = DataSet[feature_cols]  # Features
y = DataSet.STATUS  # Target variable

print("Columnas de entrada")
print(X)
print("Etiqueta(estado de la llama):")
print(y)  # Esta sería mi etiqueta

# Divido mi dataset en prueba y entrenamiento

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 80% training and 20% test

#Escalo mis variables

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creamos el objeto "red neuronal"

red = Sequential()

# Capa de entrada con 7 deuronas de entrada y 50 salidas:

red.add(Dense(
    100,                                 #Neuronas de la capa de salida
    kernel_initializer='he_uniform',
    activation='relu',                  #Función de activación
    input_shape=(7,),                   #Neuronas de entrada
))

# 1er Capa oculta

red.add(Dense(
    100,                                 #Neuronas de la capa de salida
    kernel_initializer='he_uniform',
    activation='relu',                  #Función de activación
))

# Capa de salida de neurona que me dirá si se apagó o no

red.add(Dense(
    1,                                     #Neuronas de la capa de salida
    kernel_initializer='he_uniform',
    activation='sigmoid',                  #Función de activación sigmoid para salida binaria
))

# Compilo la red que armé recién

red.compile(
    optimizer = 'adam',                 # Se requiere para usar el descenso de gradiente
    loss = 'binary_crossentropy',       # Reduce la entropía
    metrics = ['accuracy'],
)

# Para entrenar la red

historial = red.fit(X_train,y_train,
        batch_size = 10,            # Cada 10 registros actualiza los pesos
        epochs = 1000,              # Cantidad de pasadas
        )

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

y_pred = red.predict(X_test)
y_pred = np.round(y_pred)           # Como y_pred es una probabilidad, la redondeo a 0 o 1 para comparar con status

# Genero la matriz de confusión

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predicción', fontsize=18)
plt.ylabel('Reales', fontsize=18)
plt.title('Matriz de confusión', fontsize=18)
plt.show()

