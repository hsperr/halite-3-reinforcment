#!/usr/bin/env python3

import hlt
from hlt import constants
from hlt.positionals import Direction

import random
import logging
import sys
import os
import numpy as np
import time

EPSILON = float(sys.argv[1])
MODEL_PATH = sys.argv[2]
TRAINING_DATA_PATH = sys.argv[3]

TRAINING_DATA_PATH = f"{TRAINING_DATA_PATH}/{EPSILON}"
if not os.path.exists(TRAINING_DATA_PATH):
    os.mkdir(TRAINING_DATA_PATH)

timestamp = int(time.time()*1000)

game = hlt.Game()
game.ready(f"{EPSILON}-{MODEL_PATH.split('/')[2]}")
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

def create_state_array(game_map, ship):
    state = np.zeros((5, game_map.width, game_map.height))
    for rows in game_map._cells:
        for cell in rows:
            state[0, cell.position.y, cell.position.x] = cell.halite_amount/1000.0

            if cell.is_occupied:
                state[1, cell.position.y, cell.position.x] = 1 if cell.ship.owner == ship.owner else -1
                state[2, cell.position.y, cell.position.x] = cell.ship.halite_amount/1000.0

            if cell.has_structure:
                state[3, cell.position.y, cell.position.x] = 1 if cell.structure.owner == ship.owner else -1

    state[4, ship.position.y, ship.position.x] = 1
    return state


if not EPSILON == 1.0:
    import tensorflow as tf

    if os.path.exists(MODEL_PATH):
        logging.info(f"Loading {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        raise Exception(f"MODEL_PATH={MODEL_PATH} not found")

states = []
action_indices = []
rewards = []
dones = []
current_qs = []

POSSIBLE_MOVES = [Direction.North, Direction.East, Direction.South, Direction.West, Direction.Still]


shipid2shipandcell = {}


while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    command_queue = []

    my_ship_ids = [ship.id for ship in me.get_ships()]
    for shipid, (old_ship_halite, old_cell_halite) in shipid2shipandcell.items():
        if not shipid in my_ship_ids:
            #old ship is dead
            rewards.append(-1000)
            dones.append(True)
        elif me.get_ship(shipid).position == me.shipyard.position and old_ship_halite:
            feuel_cost_last_round = (old_cell_halite * 0.1)
            rewards.append(round(old_ship_halite - feuel_cost_last_round + 0.5))
            dones.append(True)
        else:
            rewards.append(-1)
            dones.append(me.get_ship(shipid).position == me.shipyard.position)

    shipid2shipandcell = {}
            
    for ship in me.get_ships():
            state = create_state_array(game_map, ship)
            states.append(state)
            if ship.halite_amount < game_map[ship.position].halite_amount * 0.1:
                move = Direction.Still # can't move anyway
                current_q = "CANTMOVE"
            elif random.random() < EPSILON:
                move = random.choice(POSSIBLE_MOVES)
                current_q = "RANDOM"
            else:
                current_q = model.predict(state.reshape(1, *state.shape))[0]
                move_index = np.argmax(current_q)
                move = POSSIBLE_MOVES[move_index]

            current_qs.append(current_q)
            action_indices.append(POSSIBLE_MOVES.index(move))
            shipid2shipandcell[ship.id] = (ship.halite_amount, game_map[ship.position].halite_amount)

            command_queue.append(ship.move(move))

    if len(my_ship_ids)<1:
        command_queue.append(me.shipyard.spawn())

    if game.turn_number == constants.MAX_TURNS:
        dones[-1] = True
        total_reward = 0
        for state, action_index, reward, done, current_q in zip(states, action_indices, rewards, dones, current_qs):
            logging.info(f"{POSSIBLE_MOVES[action_index]} | {reward} | {done} | {current_q}")

        total_reward+=sum(rewards)
        logging.info(f"TotalReward={total_reward}")

        np.savez(f"{TRAINING_DATA_PATH}/{total_reward}_{timestamp}", np.array([states, action_indices, rewards, dones]))

    game.end_turn(command_queue)
