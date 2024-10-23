#!/bin/bash

procs=36

i=25
while [ "$i" -le $procs ]; do
    python data/sample_points.py --track 0 --seed $i --list "output/track_list.hdf5" --out "output/workers/" --runs 1 --steps 15 &
    i=$(( i + 1 ))
done
