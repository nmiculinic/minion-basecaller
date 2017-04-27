FROM nvidia/cuda:8.0-cudnn5-devel

# Mostly copy/paste from tensorflow webpage
# Pick up some TF dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-setuptools \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-scipy \
        python3-sklearn \
        python \
        python-pip \
        python-numpy \
        python-matplotlib \
        python-setuptools \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        libhdf5-serial-dev \
        git \
        cmake \
        sshfs \
        autoconf \ 
        libbz2-dev \
        liblzma-dev \
        libncurses5-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# -gpu for gpu build
RUN pip3 --no-cache-dir install tensorflow git+https://github.com/tflearn/tflearn.git Pillow \
        h5py python-dotenv sigopt git+https://github.com/nmiculinic/edlib-python.git \
        slacker-log-handler dill pysam tqdm

RUN pip --no-cache-dir install -U matplotlib

WORKDIR /opt
ENV TENSORFLOW_SRC_PATH=/opt/tensorflow
ENV WARP_CTC_PATH=/opt/warp-ctc/build
ENV CUDA_HOME=/usr/local/cuda

RUN git clone https://github.com/tensorflow/tensorflow.git tensorflow
RUN git clone https://github.com/nmiculinic/warp-ctc.git warp-ctc
RUN git clone https://github.com/isovic/graphmap.git graphmap --recursive
RUN git clone https://github.com/isovic/samscripts.git samscripts
RUN git clone https://github.com/samtools/samtools
RUN git clone https://github.com/samtools/htslib
RUN git clone https://github.com/samtools/bcftools

WORKDIR /opt/graphmap
RUN make && make install

WORKDIR /opt/htslib
RUN autoheader && autoconf && ./configure 
RUN make && make install
WORKDIR /opt/samtools
RUN autoconf -Wno-syntax  && ./configure && make && make install
WORKDIR /opt/bcftools
RUN make && make install


WORKDIR /opt/warp-ctc
RUN mkdir build
WORKDIR /opt/warp-ctc/build
RUN cmake .. && make
WORKDIR /opt/warp-ctc/tensorflow_binding
RUN python3 setup.py install

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /opt/warp-ctc/build:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN mkdir /code
RUN mkdir /data
WORKDIR /code
