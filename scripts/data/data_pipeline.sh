#!/bin/bash
# This file calls the sampling script multiple times
# Change desired parameters here
# If desired, the whole process of generating training data could also be automated here.

procs=30

i=1
while [ "$i" -le $procs ]; do
    python -O scripts/data/sample_points.py --track 0 --seed $i --list "output/mpc_data/mm.hdf5" --out "output/mpc_data/workers/" --runs 25 --steps 48 &
    i=$(( i + 1 ))
done
