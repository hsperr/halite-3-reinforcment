#!/bin/sh

MODEL_NAME=$1
DATA_PATH=$2

./halite  --turn-limit 100 --replay-directory replays/ --no-timeout --no-logs -vvv --width 8 --height 8 "python3 NNBot.py 0.0 $MODEL_NAME $DATA_PATH" "python3 rule_based/CollectDataBot.py"
./halite  --turn-limit 100 --replay-directory replays/ --no-timeout --no-logs -vvv --width 8 --height 8 "python3 rule_based/CollectDataBot.py" "python3 NNBot.py 0.0 $MODEL_NAME $DATA_PATH"


ls data/single_ship/beat_rule/0.0 | wc

# TDWIN: 12
# NNWIN: 8