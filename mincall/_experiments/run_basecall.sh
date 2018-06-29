#!/usr/bin/env bash

SAM_FILE="$HOME/Desktop/eval_bc/alignment.sam"
python -m mincall -v basecall --in ~/Desktop/eval_bc/sample_ecoli_simpson/ --out ~/Desktop/eval_bc/out.fasta --model ~/Desktop/eval_bc/mincall_keras_model.save -t 8 --beam 50
graphmap align -r ~/Desktop/eval_bc/ref.fasta -d ~/Desktop/eval_bc/out.fasta -o "$SAM_FILE" -C --extcigar
# minimap2 --eqx -ax map-ont ~/Desktop/eval_bc/ref.fasta ~/Desktop/eval_bc/out.fasta > "$SAM_FILE"
python mincall/_experiments/sam_stats.py "$SAM_FILE"