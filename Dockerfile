FROM maven:3.9.9-eclipse-temurin-11 as bundle
WORKDIR /workspace
COPY bundle .
RUN mvn clean install

FROM ubuntu:24.10
RUN apt update && apt upgrade -y

RUN apt install python3 python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt install java-11-openjdk-headless -y
COPY --from=bundle /workspace/target/bundle-*-standalone.jar /workspace/bundle.jar

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt
COPY . .