FROM ubuntu:24.10
RUN apt update && apt upgrade -y

RUN apt install python3 python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt
COPY . .