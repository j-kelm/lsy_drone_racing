#!/bin/bash

procs=48

i=25
while [ "$i" -le $procs ]; do
    python -O scripts/data/sample_points.py --track 0 --seed $i --list "output/mm.hdf5" --out "output/workers/" --runs 10 --steps 48 &
    i=$(( i + 1 ))
done
