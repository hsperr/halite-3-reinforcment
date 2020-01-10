#!/usr/bin/env python3

import hlt
from hlt import constants
from hlt.positionals import Direction

import random
import logging
import sys
import os
import json
import hashlib
import numpy as np

EPSILON = float(sys.argv[1])
TABLE = sys.argv[2]

game = hlt.Game()
game.ready(f"{EPSILON}-{TABLE}")

def create_state_array(game_map, ship):
    state = np.zeros((4, game_map.width, game_map.height))
    for rows in game_map._cells:
        for cell in rows:
            state[0, cell.position.x, cell.position.y] = round(cell.halite_amount/1000.0, 2)
            state[1, cell.position.x, cell.position.y] = round((cell.ship.halite_amount+1)/1001, 2) if cell.is_occupied and cell.ship.owner == ship.owner else 0
            state[2, cell.position.x, cell.position.y] = 1 if cell.has_structure and cell.structure.owner == ship.owner else 0
    state[3, ship.position.x, ship.position.y] = 1
    return state


if os.path.exists(TABLE):
    logging.info(f"Loading {TABLE}")
    with open(TABLE, 'rb') as f:
        table = json.loads(f.read())
else:
    logging.info(f"Initializing {TABLE}")
    table = {}

states = []
action_indices = []
rewards = []
dones = []

LEARNING_RATE = 0.1
DISCOUNT = 0.95

POSSIBLE_MOVES = [Direction.North, Direction.East, Direction.South, Direction.West, Direction.Still]

def state2key(state):
    return hashlib.sha224(f"{state.tostring()}".encode("UTF-8")).hexdigest() 

def lookup_table(table, state):
    return table.setdefault(state2key(state), list(np.random.uniform(0, 1, len(POSSIBLE_MOVES)))+[0])


logging.info("Successfully1 created bot! My Player ID is {}.".format(game.my_id))

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
            dones.append(False)
            continue
        elif me.get_ship(shipid).position == me.shipyard.position and old_ship_halite:
            feuel_cost_last_round = (old_cell_halite * 0.1)
            rewards.append(round(old_ship_halite - feuel_cost_last_round + 0.5))
            dones.append(True)
        else:
            rewards.append(-1)
            dones.append(False)

    shipid2shipandcell = {}
            
    for ship in me.get_ships():
            state = create_state_array(game_map, ship)
            states.append(state)
            if random.random() < EPSILON:
                move = random.choice(POSSIBLE_MOVES)
            else:
                current_q = lookup_table(table, state)
                move_index = np.argmax(current_q[:-1])
                move = POSSIBLE_MOVES[move_index]

            action_indices.append(POSSIBLE_MOVES.index(move))
            shipid2shipandcell[ship.id] = (ship.halite_amount, game_map[ship.position].halite_amount)

            command_queue.append(ship.move(move))

    if len(my_ship_ids)<1:
        command_queue.append(me.shipyard.spawn())

    if game.turn_number == constants.MAX_TURNS:
        # logging.info(f"{len(states)} - {len(action_indices)} - {len(rewards)} - {len(dones)}")
        for state, next_state, action_index, reward, done in zip(states, states[1:], action_indices, rewards, dones):
            current_q = lookup_table(table, state)
            future_q = lookup_table(table, next_state)

            if done:
                current_q[action_index] = reward
            else:
                current_q[action_index] = LEARNING_RATE * current_q[action_index] + (1 - LEARNING_RATE) * (reward + DISCOUNT * np.max(future_q[:-1]))

            current_q[-1] += 1

            table[state2key(state)] = current_q

            logging.info(f"{state2key(state)} | {POSSIBLE_MOVES[action_index]} | {reward} | {done} | {[round(x, 2) for x in current_q]}")

        logging.info(f"Writing to {TABLE} - {len(table)} - {len(states)}")
        with open(TABLE, 'w') as f:
            f.write(json.dumps(table))

    game.end_turn(command_queue)
