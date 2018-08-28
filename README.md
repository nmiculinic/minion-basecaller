# MinCall

This is MinION DNA basecaller developed from 2016-2018, and served as mine master thesis. The whole master thesis is [attached](https://github.com/nmiculinic/minion-basecaller/blob/master/nmiculinic_ma.pdf) to this repo with the detailed introduction, problem description, related work, methodology, results and final commentary.

For a quick summary of master thesis: 

* It's a deep learning approach using a sequence to sequence models with CTC loss with CNN's. 
* No LSTM were used which leads to high speeds. 
* During performance tuning Beam-search proved the bottleneck. 
* If the Greedy search is used per-read results are slightly worse, but the read consensus is overall better.
* Whole code uses modern tensorflow and keras, with a custom pre-processing pipeline. 
* It uses simple YAML config file, validated using the voluptuous library 
* This is primarily scientific code, thus I've aimed to the best quality and software engineering practices possible within given timeline, though priority were the results. This code has been through one rewrite from scratch which improved its quality and ease of use.
* Many progress bars are implemented using tqdm library and it's possible to add graylog logging backend. I've used graylog for log management during training, much better than searching through log files.
* For further questions feel free to contact me on my email or open the issue. University one is disabled (( I've graduated )) but a private one is available on my GitHub profile.


## Setup 

* Install via `python setup.py install` or use docker images (( via makefile ))
* python3 -m mincall basecall --in <fast5 folder> --out out.fasta --model model.save
* Or via docker it's similar:
    * `docker run --rm -it nmiculinic/mincall:latest-py3 basecall ...`
    * `docker run --runtime=nvidia --rm -it nmiculinic/mincall:latest-py3-gpu basecall ...`
    GPU version required nvidia-docker installed

For all other information look at the attached master thesis (nmiculinic_ma.pdf), and the source code.

## Further work

* Perform model training on bigger dataset than E. Coli R9.4 . My estimate is the results shall be similar to other basecallers.
