#!/bin/bash
set -e

python train.py --name r4_s0 --rank 4 --seed 0
python train.py --name r64_s0 --rank 64 --seed 0

python train.py --name r16_s1 --rank 16 --seed 1
python train.py --name r16_s2 --rank 16 --seed 2

for name in r4_s0 r64_s0 r16_s1 r16_s2; do
    python evaluate.py --name $name --ckpt runs/$name/best.pt
done

python report.py --main r16_s0 --others r4_s0 r64_s0 r16_s1 r16_s2