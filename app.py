import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly_express as px



@st.cache_rsource
def carrega_modelo():
    #https://drive.google.com/file/d/1UPVJjwZ1lf9HPIiGXUVpD3fOoEn3AaJW/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1UPVJjwZ1lf9HPIiGXUVpD3fOoEn3AaJW'
    gdown.download(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image)
        st.success("Imagem carregada com sucesso")
        
        # REDIMENSIONA para 416x416 (tamanho esperado pelo modelo)
        image = image.resize((416, 416))
        
        # Converte para array e normaliza
        image = np.array(image, dtype=np.float32)
        image = image / 416.0  # Mantém a normalização do treinamento
        image = np.expand_dims(image, axis=0)
        return image
    
def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'],image) 
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['Imature', 'Mature']
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]
    
    fig = px.bar(df,y='classes', x='probabilidades (%)',  orientation='h', text='probabilidades (%)', title='Probabilidade de Classes de Doenças em Uvas')
    st.plotly_chart(fig)

def main():

    st.set_page_config(
        page_title="Classificador de Cataratas",
    )

    st.write("# Classifica OLHOS")



    #Carrega Modelo
    interpreter = carrega_modelo()

    #Carrega Imagem

    image = carrega_imagem()

    #Classifica
    if image is not None:
        previsao(interpreter, image)

if __name__ == "__main__":#faz com que a função rode
    main()