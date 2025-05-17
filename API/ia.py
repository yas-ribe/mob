#region Importações
# Dados
import pandas as pd
# SKlearn
  # Treino e teste
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
  # Modelos
from sklearn.linear_model import LinearRegression

# Base de dados
url = "https://media.githubusercontent.com/media/2025-1-NCC5/Projeto3/refs/heads/main/documentos/DataFrame/dadosIntegrados.csv"
df = pd.read_csv(url)
#endregion

# Retirando a coluna de index duplicada da importação
df.drop('Unnamed: 0', axis=1, inplace=True)

# Criando o treino e teste
x = df.drop(['Preco', 'ID_Corrida'], axis=1)
y = df['Preco']

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

treino_x, teste_x, treino_y, teste_y = train_test_split(
  scaled_x,y,test_size=0.3, random_state=42)

# Definindo o modelo
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(treino_x, treino_y)

#region Teste
'''
# Testando novas previsões
Categoria = int(input('Categoria:'))
Mes = int(input('Mes:'))
Dia = int(input('Dia:'))
Hora = float(input('Hora:'))
Final_semana = int(input('Final_semana:'))
Distancia = float(input('Distancia:'))

entrada = pd.DataFrame([{'ID_Categoria': Categoria, 'Mes': Mes, 'Dia': Dia, 'Hora': Hora, 'Final_semana':Final_semana, 'Distancia': Distancia}])
entrada_scaled = scaler.transform(entrada)
pred = modelo.predict(entrada_scaled)[0]

# Verificando qual a categoria de Uber selecionada
categoria_tipo = ''
match Categoria:
  case 0:
    categoria_tipo = 'Outros'
  case 1:
    categoria_tipo = 'UberX'
  case 2:
    categoria_tipo = 'UberBlack'
  case 3:
    categoria_tipo = 'Uber Taxi'
  case 4:
    categoria_tipo = 'Uber Executivo'
  case 5:
    categoria_tipo = 'UberFlash'

# Imprimindo os resultados
print("Resultado\n")
print(f'Dia: {Dia}')
print(f'Hora: {Hora:.0f}')
print(f'Categoria: {categoria_tipo}')
print(f'Distancia: {Distancia:.0f}')
if Final_semana == 1:
  print(f'Final de semana: Sim')
else:
  print(f'Final de semana: Não')
print(f'Preço: R${round(pred, 2)}')
print("\n------------------------------------------------------------\n")
'''
#endregion