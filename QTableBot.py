#!/usr/bin/env python3

import hlt
from hlt import constants
from hlt.positionals import Direction

import random
import logging


game = hlt.Game()
game.ready("MyPythonBot")

logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    command_queue = []

    for ship in me.get_ships():
            move = random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])
            command_queue.append(ship.move(move))

    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)

