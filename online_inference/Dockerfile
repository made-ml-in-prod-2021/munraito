FROM python:3.6-slim-stretch
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY src app/src
COPY model app/model
COPY app.py app/app.py

WORKDIR /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]