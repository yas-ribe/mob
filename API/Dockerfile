
FROM python:3.13-slim
RUN pip install pipenv
WORKDIR /app
COPY . /app
RUN pipenv install --deploy --ignore-pipfile
EXPOSE 5000
CMD ["pipenv", "run", "python", "api.py"] 