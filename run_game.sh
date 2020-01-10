#!/bin/sh


./halite --seed 123 --replay-directory replays/ --turn-limit 10 --no-logs -vvv --width 8 --height 8 "python3 QTableBot.py 0.0 table_left" "python3 DoNothingBot.py"