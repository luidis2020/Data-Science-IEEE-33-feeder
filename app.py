import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv("model/data.csv")


# função para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop("Voltage_bus32",axis=1)
    y = data["Voltage_bus32"]
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
    rf_regressor.fit(x, y)
    return rf_regressor

# criando um dataframe
data = get_data()

# treinando o modelo
model = train_model()

# título
st.title("Data App - Forescasting Voltage (p.u) bus 32")

# subtítulo
st.markdown("This app forecast voltage values using Machine Learning for the problem of forescating voltages.")

# verificando o dataset
st.subheader("Select a set of information")

# atributos para serem exibidos por padrão
defaultcols = ["Voltage_bus32"]

# defindo atributos a partir do multiselect
cols = st.multiselect("Information", data.columns.tolist(), default=defaultcols)

# exibindo os top 10 registro do dataframe
st.dataframe(data[cols].head(10))


st.subheader("Probability Density Function Voltages (p.u)")

# definindo a faixa de valores
faixa_valores = st.slider("Range Voltage", float(data.Voltage_bus32.min()), 150., (10.0, 100.0))

# filtrando os dados
dados = data[data['Voltage_bus32'].between(left=faixa_valores[0],right=faixa_valores[1])]

# plot a distribuição dos dados
f = px.histogram(dados, x="Voltage_bus32", nbins=100, title="Probability Density Function")
f.update_xaxes(title="Voltage_bus32")
f.update_yaxes(title="PDF")
st.plotly_chart(f)


st.sidebar.subheader("Select values from power injected main feeder")

# mapeando dados do usuário para cada atributo
MFPA = st.sidebar.number_input("Active Power Main feeder", value=data.Pcal_bus1.mean())
MFPR = st.sidebar.number_input("Reactive Power Main Feeder", value=data.Qcal_bus1.mean())
ITER = st.sidebar.number_input("Iteration", value=data.Iteration.mean())

# inserindo um botão na tela
btn_predict = st.sidebar.button("Forecast Voltage Bus 32")

# verifica se o botão foi acionado
if btn_predict:
    result = model.predict([[MFPA,MFPR,ITER]])
    st.subheader("Voltage value forecast bus 32:")
    result = str(result)+"p.u "
    #result = "p.u "+str(round(result[0]*10,2))
    st.write(result)