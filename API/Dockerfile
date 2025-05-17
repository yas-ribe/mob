# Use imagem oficial do Python
FROM python:3.13-slim

# Instala pipenv
RUN pip install pipenv

# Cria diretório de trabalho
WORKDIR /app

# Copia arquivos do projeto
COPY . /app

# Instala dependências
RUN pipenv install --deploy --ignore-pipfile

# Expõe a porta da API
EXPOSE 5000

# Comando para rodar a API
CMD ["pipenv", "run", "python", "api.py"] 