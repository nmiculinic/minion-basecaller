# MinCall

* Setup:
    * Install via `python setup.py install` or use docker images (( via makefile ))
    * python3 -m mincall basecall --in <fast5 folder> --out out.fasta --model model.save
    * Or via docker it's similar:
        * `docker run --rm -it nmiculinic/mincall:latest-py3 basecall ...`
        * `docker run --runtime=nvidia --rm -it nmiculinic/mincall:latest-py3-gpu basecall ...`
        GPU version required nvidia-docker installed

For all other informations look at the attached master thesis (nmiculinic_ma.pdf) , and the source code.


## TODO

* Add best current model. 
* In about month-2 (( end of Summer 2018 )) I'll rerun training pipeline on bigger dataset
