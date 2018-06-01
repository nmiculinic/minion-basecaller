FROM ubuntu:16.04

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        python3.6 \
        python3.6-dev \
        python3-distutils \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN python3.6 -m pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        tensorflow \
        && \
    python3.6 -m ipykernel.kernelspec

# TensorBoard
EXPOSE 6006

WORKDIR /
RUN mkdir /code
RUN mkdir /data
WORKDIR /opt
COPY requirements.txt requirements.txt.bak
RUN cat requirements.txt.bak | grep -v tensorflow > requirements.txt && rm requirements.txt.bak
RUN python3.6 -m pip --no-cache-dir install -r requirements.txt && rm requirements.txt
WORKDIR /code
ENV PYTHONPATH=/code
COPY . .
RUN ["python3.6", "setup.py", "install"]
ENTRYPOINT ["python3.6", "-m", "mincall"]
