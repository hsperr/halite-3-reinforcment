import sys
import json
import subprocess

MODEL1_PATH = sys.argv[1]

command = './halite#' \
          '--replay-directory replays/#' \
          '--no-timeout#' \
          '--no-logs#-v#' \
          '--no-replay#' \
          f'--width 8#' \
          f'--results-as-json#' \
          f'--height 8#' \
          f'python3 NNBot.py 0.0 {MODEL1_PATH} data/single_ship/beat_rule#' \
          f'python3 rule_based/CollectDataBot.py'

wins = {MODEL1_PATH: 0, "TheDorian": 0}
total = 0
GAMES = 50
for i in range(GAMES):
    result = subprocess.run(command.split('#'), stdout=subprocess.PIPE)
    as_json = json.loads(result.stdout.decode("utf-8"))

    if as_json['stats']['0']['rank'] == 1:
        wins[MODEL1_PATH]+=1
    else:
        wins["TheDorian"]+=1

    print(wins)

