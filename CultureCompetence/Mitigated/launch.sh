#!/bin/sh

# st in standards = [0, 1] param 1
# t in times=[0,...,9] 
# l in lambs=[-1,0, ..., 25] param 2
# per in percents=[0.05, 0.1] param 3
# c in culture=[0,1,2] param 4
# adv in adv/aug=[0,1,2,3] param 5

for st in 0 1
do
for t in 0 1 2 3 4 5 6 7 8 9
do
for l in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13
do
for per in 0.05 0.1
do
for c in 0 1 2 
do
for adv in 0 1 2 3 
do
python3 general.py $st $l $per $c $adv
done
done
done
done
done
done