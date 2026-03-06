FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/learn data/weights data/visualisation samples
COPY data/learn/sample_Iris.csv samples/sample_Iris.csv
COPY data/learn/sample_colors.csv samples/sample_colors.csv

EXPOSE 8000

CMD ["python", "main.py"]
