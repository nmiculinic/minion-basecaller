FROM scivm/scientific-python-2.7

WORKDIR /opt
RUN git clone https://github.com/nanoporetech/nanonet.git
WORKDIR /opt/nanonet
RUN git checkout tags/v2.0.0
# RUN python2 setup.py install
