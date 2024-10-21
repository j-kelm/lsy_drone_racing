#!/bin/bash

procs=8

i=1
while [ "$i" -le $procs ]; do
    python data/sample_points.py --track 0 --seed $i --list "output/track_list.hdf5" --out "output/workers/" --runs 1 --steps 1 &
    i=$(( i + 1 ))
done
