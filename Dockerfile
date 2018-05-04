FROM python:3

ENV TENSORFLOW_SRC_PATH=/opt/tensorflow
ENV WARP_CTC_PATH=/opt/warp-ctc/build
ENV CUDA_HOME=/usr/local/cuda

# RUN git clone https://github.com/tensorflow/tensorflow.git tensorflow
# RUN git clone https://github.com/nmiculinic/warp-ctc.git warp-ctc
# RUN git clone https://github.com/isovic/graphmap.git graphmap --recursive
# RUN git clone https://github.com/isovic/samscripts.git samscripts
# RUN git clone https://github.com/samtools/samtools
# RUN git clone https://github.com/samtools/htslib
# RUN git clone https://github.com/samtools/bcftools

WORKDIR /opt

#WORKDIR /opt/warp-ctc
#RUN mkdir build
#WORKDIR /opt/warp-ctc/build
#RUN cmake .. && make
#WORKDIR /opt/warp-ctc/tensorflow_binding
#
#RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
#RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH python3 setup.py install
#RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

# For CUDA profiling, TensorFlow requires CUPTI.
# ENV LD_LIBRARY_PATH /opt/warp-ctc/build:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH



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
