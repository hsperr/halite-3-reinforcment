#!/bin/sh

MODEL_NAME=$1
DATA_PATH=$2

./halite --seed 123 --turn-limit 100 --replay-directory replays/ --no-timeout --no-logs -vvv --width 8 --height 8 "python3 NNBot.py 0.1 $MODEL_NAME $DATA_PATH" "python3 NNBot.py 0.1 $MODEL_NAME $DATA_PATH"
