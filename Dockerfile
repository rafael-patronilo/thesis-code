#required for running the justifier
FROM ubuntu:24.10 AS buddy
RUN apt update && apt upgrade -y && \
    apt install -y openjdk-8-jdk-headless \
        build-essential automake autoconf libtool
COPY ./dependencies/javabdd /javabdd
WORKDIR /javabdd
RUN make

FROM ubuntu:24.10 AS justifier
RUN apt update && apt upgrade -y && \
    apt install -y openjdk-8-jdk-headless
COPY ./dependencies/justifier/src /justifier/src
COPY ./dependencies/justifier/Justifier_original.jar /justifier/Justifier_original.jar
WORKDIR /justifier/src
RUN mkdir ../bin
RUN javac -cp ../Justifier_original.jar:. -d ../bin *.java **/*.java
WORKDIR /justifier/temp
RUN jar xf ../Justifier_original.jar
RUN cp --remove-destination -r ../bin/* .
RUN jar cfe ../Justifier.jar Main *

FROM ubuntu:24.10
RUN apt update && apt upgrade -y && \
    apt install -y openjdk-8-jdk-headless \
        python3 python3-pip -y

COPY --from=buddy /javabdd/libbuddy.so /usr/lib/libbuddy.so

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt

COPY --from=justifier /justifier/Justifier.jar .

COPY . .
