import json
from flask import Flask, jsonify, request
import pandas as pd
import ia
import datetime
from datetime import datetime

# Importações para o ORS
import os
import requests
import json


app = Flask(__name__)

dados = []

nextId = 0

ORS_API_KEY = '5b3ce3597851110001cf6248b246d8e65ded4136b7f9a45e60d96167'

@app.route('/dados', methods=['GET'])
def get_data():
 return jsonify(dados)

def data_is_valid(dados):
  required_keys = ['lonOri', 'latOri', 'lonDes', 'latDes']

  if all(key in dados for key in required_keys):
    return True
  return False

@app.route('/calcular', methods=['POST'])
def create_data():
  global nextId
  dado = json.loads(request.data)
  if not data_is_valid(dado):
    return jsonify({ 'erro': 'propriedades inválidas.' }), 400
  
  lonOri = dado['lonOri']
  latOri = dado['latOri']
  lonDes = dado['lonDes']
  latDes = dado['latDes']
  
  # Enviando dados de endereço para a ORS para calcular distância e tempo
  headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
  }
  call = requests.get(f'https://api.openrouteservice.org/v2/directions/driving-car?api_key={ORS_API_KEY}&start={lonOri},{latOri}&end={lonDes},{latDes}', headers=headers)

  distancia = float(round(call.json()['features'][0]['properties']['segments'][0]['distance']/1000, 2))
  duracao = float(round(call.json()['features'][0]['properties']['segments'][0]['duration']/60, 2))
  
  modelo = ia.modelo
  categoria = 1
  mes = datetime.now().month
  dia = datetime.now().weekday()
  hora = datetime.now().hour
  minuto = datetime.now().minute
  segundo = datetime.now().second
  if datetime.now().weekday() == 5 or datetime.now().weekday() == 6:
    final_semana = 1
  else:
    final_semana = 0

  entrada = pd.DataFrame([{'ID_Categoria': categoria, 'Mes': mes, 'Dia': dia, 'Minuto': minuto, 'Segundo': segundo,
                           'Hora': hora, 'Final_semana': final_semana, 'Distancia': distancia, 'Tempo': duracao}])
  
  scaler = ia.scaler
  entrada_scaled = scaler.transform(entrada)
  pred = round(modelo.predict(entrada_scaled)[0], 2)

  dado['id'] = nextId
  nextId += 1

  dados.append(dado)

  return jsonify(pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)