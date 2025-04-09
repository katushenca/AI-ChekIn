FROM python:3.10

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY src/ /app

CMD ["python", "/app/app.py"]
