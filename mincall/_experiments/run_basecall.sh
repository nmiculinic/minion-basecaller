#!/usr/bin/env bash

python -m mincall -v basecall --in ~/Desktop/eval_bc/single_fast5/ --out ~/Desktop/eval_bc/out.fasta --model ~/Desktop/eval_bc/mincall_keras_model.save -t 8 --beam 50
graphmap align -r ~/Desktop/eval_bc/ref.fasta -d ~/Desktop/eval_bc/out.fasta -o ~/Desktop/eval_bc/alignment.sam -C --extcigar
#minimap2 --eqx -ax map-ont ~/Desktop/eval_bc/ref.fasta ~/Desktop/eval_bc/out.fasta > ~/Desktop/eval_bc/alignment.sam
python mincall/_experiments/sam_stats.py ~/Desktop/eval_bc/alignment.sam