#!/bin/sh

./halite --seed 123 --replay-directory replays/ --turn-limit 10 --no-logs -vvv --width 16 --height 16 "python3 QTableBot.py $2 $1" "python3 DoNothingBot.py"