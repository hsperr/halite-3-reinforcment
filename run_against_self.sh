#!/bin/sh

MODEL_NAME=$1
MODEL_NAME2=$2
DATA_PATH=$3

./halite --turn-limit 100 --replay-directory replays/ --no-timeout --results-as-json --no-logs -vvv --width 8 --height 8 "python3 NNBot.py 0.0 $MODEL_NAME $DATA_PATH" "python3 NNBot.py 0.0 $MODEL_NAME2 $DATA_PATH"
