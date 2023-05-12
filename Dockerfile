FROM ubuntu:22.04

ENV TZ="America/New_York"
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

WORKDIR /opt/app
ADD . /opt/app

RUN apt update
RUN apt install -y python3 python3-pip
RUN python3 -m pip install -r requirements.txt
