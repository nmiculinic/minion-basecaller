FROM minion
MAINTAINER neven.miculinic@gmail.com
LABEL version=9947283
VOLUME ["/data"]

COPY mincall /opt/python/mincall
COPY models/residual_deep_prod_17466_9947283 /model

RUN chmod -R 777 /opt/python/mincall
RUN chmod -R 777 /model
ENV PYTHONPATH=/opt/python
CMD python3 /opt/python/mincall/models/residual_deep_albacore.py basecall /model /data
