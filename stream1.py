import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

modelo = load_model('meu-modelo-para-charges')





paginas = ['Home','Modelagem', 'Sobre']

pagina = st.sidebar.selectbox('Escolha sua página:', paginas)

if pagina == 'Home':
    st.title('Meu primeiro StreamLit Web App')
    st.subheader('by Eduardo')

    n = st.slider('Entre com um numero:',1,10,3)
    frase = 'O quadrado do valor selecionado é: {}'.format(n**2)

    st.subheader(frase)



if pagina == 'Modelagem':

    st.subheader('Entre com as variaveis para fazer a previsao do seguro:')
    st.markdown('---')

    # st.subheader('Modelagem under construction...')
    idade = st.number_input('Idade', 18,65,30)
    sexo = st.selectbox('Sexo', ['male','female'])
    imc = st.number_input('Indice de Massa Corporal', 15,54,24)
    criancas = st.selectbox("Quantidade de filhos",[0,1,2,3,4,5])
    fumante = st.selectbox("É fumante?", ['yes','no'])
    regiao = st.selectbox('Regiao em que mora', ['southeast', 'southwest','northeast','southwest'])

    dados0 = {'age':[idade], 'sex':[sexo], 'bmi':[imc], 'children':[criancas], 'smoker':[fumante], 'region':[regiao]}
    dados = pd.DataFrame(dados0)

    st.markdown('---')





    if st.button('Executar modelo'):
        pred = float(predict_model(modelo, data=dados)['Label'].round(2))
        #st.write(pred)
        saida = 'O valor predito é:  {}'.format(pred)
        st.subheader(saida)

if pagina == 'Sobre':
    st.subheader('Sobre under construction...')
