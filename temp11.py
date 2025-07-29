import pandas as pd
import seaborn as sb
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Título de la aplicación
st.title("Predicción de Aprobación de Estudiantes con Árbol de Decisión")
st.markdown("Este modelo usa notas: Parciales, Proyecto y Examen Final para predecir si un estudiante aprobaría la materia.")

# Cargar los datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("estudiantes_notas_finales.csv")

# Cargar el dataset
df = cargar_datos()

# Mostrar los primeros registros
st.subheader("Datos cargados")
st.write(df.head())

# Validar que las columnas necesarias existan
required_columns = ["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final", "Nota_Final", "Aprobado"]
if not all(col in df.columns for col in required_columns):
    st.error("El dataset no contiene todas las columnas necesarias.")
    st.stop()

# Mostrar gráfico de distribución de notas (media de cada nota)
st.subheader("Distribución de notas")
st.bar_chart(df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final", "Nota_Final"]].mean())

# Dividir las variables predictoras y la variable objetivo
X = df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final"]]
y = df["Aprobado"]

# Validar que y solo tenga dos clases
if y.nunique() > 2:
    st.error("La variable 'Aprobado' debe ser binaria (Sí/No).")
    st.stop()

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Entrenar modelo
modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
modelo.fit(X_train, y_train)

# Predicción y evaluación
y_pred = modelo.predict(X_test)
st.subheader("Evaluación del Modelo")
st.text(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")

# Visualizar el árbol
st.subheader("Visualización del Árbol de Decisión")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(modelo, feature_names=X.columns, class_names=["No", "Sí"], filled=True, rounded=True, fontsize=10)
st.pyplot(fig)

# Interfaz interactiva para predicción personalizada
st.subheader("¿Aprobaría este estudiante?")

with st.form("Formulario de Predicción de Notas"):
    p1 = st.number_input("Primer Parcial", min_value=0.0, max_value=100.0, value=50.0)
    p2 = st.number_input("Segundo Parcial", min_value=0.0, max_value=100.0, value=50.0)
    proy = st.number_input("Proyecto", min_value=0.0, max_value=100.0, value=50.0)
    ef = st.number_input("Examen Final", min_value=0.0, max_value=100.0, value=50.0)
    submitted = st.form_submit_button("Predecir")

if submitted:
    datos_nuevos = pd.DataFrame([[p1, p2, proy, ef]], columns=X.columns)
    prediccion = modelo.predict(datos_nuevos)[0]
    st.success(f"Resultado: {'Aprobado' if prediccion.lower() == 'sí' or prediccion.lower() == 'si' else 'Reprobado'}")
