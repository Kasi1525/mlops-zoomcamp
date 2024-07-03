FROM python:3.10.12-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app


RUN pipenv install --system --deploy

COPY ["source_data_preparation.ipynb", "./"]
COPY ["Registry_model", "./"]
COPY ["load_module.ipynb", "./"]
COPY ["mlflow.db", "./"]
COPY ["mlruns/", "./mlruns/"]
COPY ["evidently_monitoring.ipynb/", "./"]

RUN mkdir models

EXPOSE 5011

#ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5011", "app:app" ]