FROM python:3.8-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /workspace

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# RUN apt-get update && apt-get install -y libgdal-dev && gdal-bin
