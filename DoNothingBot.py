#!/usr/bin/env python3

import hlt
from hlt import constants
from hlt.positionals import Direction

import random
import logging


game = hlt.Game()
game.ready("DoNothingBot")

logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    command_queue = []
    game.end_turn(command_queue)

