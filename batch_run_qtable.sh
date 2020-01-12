#!/bin/sh
TABLE=$1
EPSILON=$2

for i in {1..500000}
do
  ./halite --seed 123 --turn-limit 10 --replay-directory replays/ --no-timeout --no-replay --no-logs -v --width 16 --height 16 "python3 QTableBot.py $2 $1" "python3 DoNothingBot.py"
  echo $i
  ls -lh ./$1
done
