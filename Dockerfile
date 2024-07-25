FROM python:3.11.9

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    libhdf5-dev

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]