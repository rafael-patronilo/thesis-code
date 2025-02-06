FROM ubuntu:24.10 as GIT
RUN apt update && apt upgrade -y
RUN apt install -y git
WORKDIR /repositories
RUN git clone https://github.com/vscosta/cudd.git
RUN git clone https://bitbucket.org/machinelearningunife/javabdd-osgi.git

FROM ubuntu:24.10
RUN apt update && apt upgrade -y

RUN apt install -y openjdk-8-jdk-headless


RUN apt install -y build-essential automake autoconf libtool
#
#COPY --from=GIT /repositories/cudd /temp/cudd
#WORKDIR /temp/cudd
#RUN autoreconf -f -i && \
#    ./configure && \
#    make && \
#    make install

COPY ./dependencies /workspace/dependencies

WORKDIR /workspace/dependencies/javabdd
RUN make
RUN cp libbuddy.so /usr/lib



RUN apt install python3 python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python


WORKDIR /workspace
COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt
COPY . .
RUN mv /workspace/bin/* .