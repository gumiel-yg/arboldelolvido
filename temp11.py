import pandas as pd
import streamlit as st

st.title("Cargar archivo de notas")

archivo = st.file_uploader("Sube el archivo CSV con las notas", type=["csv"])

@st.cache_data
def cargar_datos(archivo):
    return pd.read_csv(archivo)

if archivo is not None:
    df = cargar_datos(archivo)
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())
else:
    st.warning("Por favor, sube un archivo CSV.")
