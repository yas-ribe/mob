import psutil
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Função para coletar dados do sistema
def coletar_dados():
    cpu = psutil.cpu_percent(interval=1)  
    memoria = psutil.virtual_memory().percent  
    disco = psutil.disk_usage('/').percent  
    net_io = psutil.net_io_counters()
    bytes_sent = net_io.bytes_sent 
    bytes_received = net_io.bytes_recv  
    return cpu, memoria, disco, bytes_sent, bytes_received

# Armazenando os dados em listas
dados = []

# Coletando dados por 1h
print("Iniciando a coleta de dados... Isso pode levar até 1 hora!")
for i in range(3600):  # 3600 iterações, ou 3600 segundos (1h)
    cpu, memoria, disco, bytes_sent, bytes_received = coletar_dados()
    dados.append([cpu, memoria, disco, bytes_sent, bytes_received])
    
    if i % 60 == 0:  # Mostrar progresso a cada 60 segundos (1 minuto)
        print(f"Coletados dados por {i // 60} minuto(s)...")
    
    time.sleep(1)

# Criando um DataFrame para armazenar os dados coletados
print("Coleta de dados concluída. Criando o DataFrame...")
df = pd.DataFrame(dados, columns=['CPU', 'Memória', 'Disco', 'Bytes Enviados', 'Bytes Recebidos'])

# Salvar os dados em um arquivo CSV para futura análise
df.to_csv('dados_monitoramento_1h.csv', index=False)
print("Dados salvos em 'dados_monitoramento_1h.csv'.")

# Carregar os dados coletados
df = pd.read_csv('dados_monitoramento_1h.csv')

# --- Regressão Linear (Previsão de Uso de CPU) ---
print("Iniciando a análise de Regressão Linear...")
X = df[['Memória', 'Disco', 'Bytes Enviados', 'Bytes Recebidos']]  
y = df['CPU']  

# Dividindo os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = modelo.predict(X_test)

# Calculando a acurácia (R²) e o erro médio quadrático (MSE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R²: {r2:.4f}")
print(f"Erro Médio Quadrático (MSE): {mse:.4f}")

# Plotando as previsões vs valores reais
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Previsão de Uso de CPU vs. Valores Reais")
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")
plt.grid(True)
plt.savefig('regressao_linear_cpu_1h.png')
plt.close()
print("Gráfico de Regressão Linear salvo como 'regressao_linear_cpu_1h.png'.")

# --- Detecção de Anomalias (Isolation Forest) ---
print("Iniciando a análise de Detecção de Anomalias...")
modelo_anomalias = IsolationForest(contamination=0.1) 

# Treinando o modelo com os dados (excluindo o target 'CPU' por enquanto)
dados_analisados = df[['Memória', 'Disco', 'Bytes Enviados', 'Bytes Recebidos']]
modelo_anomalias.fit(dados_analisados)

# Prevendo as anomalias
anomalias = modelo_anomalias.predict(dados_analisados)

# Anomalias detectadas (1 = normal, -1 = anômalo)
df['Anomalia'] = anomalias

# Plotando as anomalias detectadas
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['CPU'], c=df['Anomalia'], cmap='coolwarm', label='Anomalias')
plt.title("Detecção de Anomalias no Uso de CPU")
plt.xlabel("Índice")
plt.ylabel("Uso de CPU (%)")
plt.legend()
plt.grid(True)
plt.savefig('deteccao_anomalias_1h.png')
plt.close()
print("Gráfico de Detecção de Anomalias salvo como 'deteccao_anomalias_1h.png'.")

# --- Clustering (K-Means) ---
print("Iniciando a análise de Clustering com K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(dados_analisados)

# Plotando os clusters
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['CPU'], c=df['Cluster'], cmap='viridis', label='Clusters')
plt.title("Clusters de Uso de CPU")
plt.xlabel("Índice")
plt.ylabel("Uso de CPU (%)")
plt.legend()
plt.grid(True)
plt.savefig('clustering_kmeans_1h.png')
plt.close()
print("Gráfico de Clustering salvo como 'clustering_kmeans_1h.png'.")

# --- Gráfico de Histórico de Uso ---
print("Gerando o gráfico de Histórico de Uso de Recursos...")
plt.figure(figsize=(12, 8))
plt.plot(df['CPU'], label="Uso de CPU (%)", color='blue')
plt.plot(df['Memória'], label="Uso de Memória (%)", color='green')
plt.plot(df['Disco'], label="Uso de Disco (%)", color='red')
plt.title("Histórico de Uso de Recursos (1 hora)")
plt.xlabel("Tempo (em segundos)")
plt.ylabel("Percentual de Uso (%)")
plt.legend()
plt.grid(True)
plt.savefig('historico_uso_recursos_1h.png')
plt.close()
print("Gráfico de Histórico de Uso de Recursos salvo como 'historico_uso_recursos_1h.png'.")

# Mensagem final
print("Análise concluída! Todos os gráficos foram salvos com sucesso.")

