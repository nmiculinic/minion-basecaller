#!/usr/bin/env bash

python -m mincall basecall --in ~/Desktop/eval_bc/sample_ecoli_simpson --out ~/Desktop/eval_bc/out.fasta --model ~/Desktop/eval_bc/mincall_keras_model.save
graphmap align -r ~/Desktop/eval_bc/ref.fasta -d ~/Desktop/eval_bc/out.fasta -o ~/Desktop/eval_bc/alignment.sam -C --extcigar
python mincall/_experiments/sam_stats.py ~/Desktop/eval_bc/alignment.sam