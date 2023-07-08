FROM python:latest

WORKDIR /usr/src/app

COPY . .

RUN apt-get update \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && python3 -m pip install -r requirements.txt

CMD python3 main.py