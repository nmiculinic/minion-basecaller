FROM ubuntu:18.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        && \
    python3 -m ipykernel.kernelspec

# Install TensorFlow CPU version from central repo
RUN pip3 --no-cache-dir install tensorflow

# TensorBoard
EXPOSE 6006

WORKDIR /
RUN mkdir /code
RUN mkdir /data
WORKDIR /opt
COPY requirements.txt requirements.txt.bak
RUN cat requirements.txt.bak | grep -v tensorflow > requirements.txt && rm requirements.txt.bak
RUN pip3 --no-cache-dir install -r requirements.txt && rm requirements.txt
WORKDIR /code
ENV PYTHONPATH=/code
ENTRYPOINT ["python3", "-m", "mincall"]
COPY . .
