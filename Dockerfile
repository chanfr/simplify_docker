FROM ubuntu:18.04

RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install -y aptitude
RUN apt-get update && aptitude install  -y \
    git \
    locate\
    python python-pip python-setuptools libglib2.0-dev libsm-dev libxrender-dev libxext-dev sudo

RUN updatedb
RUN pip install --upgrade pip

RUN apt-get install -y sudo
RUN useradd -ms /bin/bash docker && echo docker:docker | chpasswd && echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers




EXPOSE 5000

WORKDIR /home/docker
USER docker


RUN git clone https://github.com/chanfr/simplify_docker

RUN cd simplify_docker

RUN sudo pip install -r requirements.txt

CMD python server.py --lenet=True
