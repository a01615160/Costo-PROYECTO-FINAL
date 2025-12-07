import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción del Costo  ''')
st.image("costito.jpg", caption="Predicción del peso de una persona.")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Presupuesto = st.number_input('Presupuesto:', min_value=0.0, max_value=10000.0, value = 0.0)
  Tiempo = st.number_input('Tiempo (Minutos):',  min_value=0, max_value=1, value = 0)
  Tipo = st.number_input('Tipo de actividad (Académico:0, Transporte:1, Alimentos/Salud:2, Ahorro inversión:3, Entretenimiento/Ocio:4, Ejercicio/Deporte:5):', min_value=0, max_value=5, value = 0)
  Momento = st.number_input('Momento del día (Mañana:0, Tarde:1, Noche:2):', min_value=0, max_value=2, value = 0)
  Personas = st.number_input('Cantidad de personas:', min_value=0, max_value=100, value = 0)

  user_input_data = {'Presupuesto': Presupuesto,
                     'Tiempo invertido': Tiempo,
                     'Tipo': Tipo,
                     'Momento': Momento,
                     'No. de personas': Personas,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('GASTOS_df.csv', encoding='latin-1')
X = datos.drop(columns='Costo')
y = datos['Costo']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1615160)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['Presupuesto'] + b1[1]*df['Tiempo invertido'] + b1[2]*df['Tipo'] + b1[3]*df['Momento'] + b1[4]*df['No. de personas']

st.subheader('Cálculo del costo')
st.write('El costo es ', prediccion)
