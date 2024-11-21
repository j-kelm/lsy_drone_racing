#!/bin/bash

procs=32

i=1
while [ "$i" -le $procs ]; do
    python -O scripts/data/sample_points.py --track 0 --seed $i --list "output/mm.hdf5" --out "output/workers/" --runs 1 --steps 12 &
    i=$(( i + 1 ))
done
