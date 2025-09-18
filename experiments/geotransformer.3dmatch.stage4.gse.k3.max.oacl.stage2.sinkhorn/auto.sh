#!bin/bash

for var in `seq 1 100`;
do
    python3 eval.py --benchmark=3DMatch --method ransac --num_corr 5000
done