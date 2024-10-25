#!/bin/bash

procs=12

i=1
while [ "$i" -le $procs ]; do
    python scripts/data/sample_points.py --track 0 --seed $i --list "output/mm.hdf5" --out "output/workers/" --runs 5 --steps 15 &
    i=$(( i + 1 ))
done
