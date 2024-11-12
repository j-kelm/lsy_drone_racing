#!/bin/bash

procs=2

i=1
while [ "$i" -le $procs ]; do
    python scripts/data/sample_points.py --track 0 --seed $i --list "output/mm.hdf5" --out "output/workers/" --runs 1 --steps 1 &
    i=$(( i + 1 ))
done
