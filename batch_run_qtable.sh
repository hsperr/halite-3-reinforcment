#!/bin/sh


for i in {1..500000}
do
  ./halite --seed 123 --turn-limit 10 --replay-directory replays/ --no-timeout --no-replay --no-logs -v --width 8 --height 8 "python3 QTableBot.py $1 table_left" "python3 DoNothingBot.py"
  echo $i
  ls -lh ./table*
done
