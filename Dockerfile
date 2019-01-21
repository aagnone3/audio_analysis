FROM continuumio/anaconda

RUN apt update --upgrade
RUN apt install -y \
    portaudio19-dev \
    python3 \
    python3-pip


RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
