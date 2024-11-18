#!/bin/bash

procs=37

i=17
while [ "$i" -le $procs ]; do
    python scripts/data/sample_points.py --track 0 --seed $i --list "output/mm.hdf5" --out "output/workers/" --runs 1 --steps 10 &
    i=$(( i + 1 ))
done
