#!/usr/bin/env python3

import hlt
from hlt import constants
from hlt.positionals import Direction, Position

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
game.ready(f"{EPSILON}-{MODEL_PATH.split('/')[1]}")
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

def create_state_array(game_map, ship):
    state = np.zeros((game_map.width, game_map.height, 6))
    for rows in game_map._cells:
        for cell in rows:
            state[cell.position.y, cell.position.x, 0] = cell.halite_amount/1000.0

            if cell.is_occupied:
                state[cell.position.y, cell.position.x, 1] = 1 if cell.ship.owner == ship.owner else -1
                state[ cell.position.y, cell.position.x, 2] = cell.ship.halite_amount/1000.0

            if cell.has_structure:
                state[cell.position.y, cell.position.x, 3] = 1 if cell.structure.owner == ship.owner else -1

    state[ship.position.y, ship.position.x, 4] = 1
    state[:, :, 5] = (constants.MAX_TURNS - game.turn_number) / constants.MAX_TURNS
    logging.info(f"max={constants.MAX_TURNS}, turn={game.turn_number}")
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
shipid2move = {}


while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    command_queue = []

    my_ship_ids = [ship.id for ship in me.get_ships()]
    for shipid, (old_ship_halite, old_cell_halite) in shipid2shipandcell.items():
        if not shipid in my_ship_ids:
            #old ship is dead
            rewards.append(-2)
            dones.append(True)
        elif me.get_ship(shipid).position == me.shipyard.position and old_ship_halite:
            feuel_cost_last_round = (old_cell_halite * 0.1)
            rewards.append(round(old_ship_halite - feuel_cost_last_round + 0.5))
            dones.append(True)
        elif me.get_ship(shipid).position == me.shipyard.position:
            rewards.append(-100)
            dones.append(True)
        else:
            # rewards.append(ship.halite_amount-old_ship_halite)
            rewards.append(-1)
            dones.append(me.get_ship(shipid).position == me.shipyard.position)

        logging.info(f"reward, {ship.id}, {rewards[-1]}, {dones[-1]}")

    shipid2shipandcell = {}
            
    for ship in me.get_ships():
            state = create_state_array(game_map, ship)
            states.append(state)
            if ship.halite_amount < game_map[ship.position].halite_amount * 0.1:
                move = Direction.Still # can't move anyway
                move_index = 4
                current_q = "CANTMOVE"
            elif random.random() < EPSILON:
                move = random.choice(POSSIBLE_MOVES)
                move_index = POSSIBLE_MOVES.index(move)
                current_q = "RANDOM"
            else:
                current_q = model.predict(state.reshape(1, *state.shape))[0]
                move_index = np.argmax(current_q)
                move = POSSIBLE_MOVES[move_index]

            current_qs.append(current_q)
            action_indices.append(POSSIBLE_MOVES.index(move))
            shipid2shipandcell[ship.id] = (ship.halite_amount, game_map[ship.position].halite_amount)

            logging.info(f"shipid={ship.id}, hlt={ship.halite_amount}, move={move_index}, pred={current_q}")
            logging.info(f"state={np.rollaxis(state, 2, 0)}")

            shipid2move[ship.id] = move

            command_queue.append(ship.move(move))

    if len(my_ship_ids)<1:
    # if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    if game.turn_number == constants.MAX_TURNS:
        for ship in me.get_ships():
            move = shipid2move[ship.id]
            if ship.position + Position(*move) == me.shipyard.position:
                dones.append(True)
                rewards.append(round(ship.halite_amount - game_map[ship.position].halite_amount * 0.1 + 0.5))
            else:
                dones.append(True)
                rewards.append(-1)

        total_reward = 0
        logging.info(f"{rewards}, {dones}")
        logging.info(f"{len(states)}, {len(action_indices)}, {len(rewards)}, {len(dones)}, {len(current_qs)}")
        for state, action_index, reward, done, current_q in zip(states, action_indices, rewards, dones, current_qs):
            logging.info(f"{POSSIBLE_MOVES[action_index]} | {reward} | {done} | {current_q}")

        total_reward+=sum(rewards)
        logging.info(f"TotalReward={total_reward}")

        # other_players = [p for pid, p in game.players.items() if pid != game.my_id]
        # if game.me.halite_amount > other_players[0].halite_amount:
        np.savez(f"{TRAINING_DATA_PATH}/{game.me.halite_amount}_{game.my_id}_{timestamp}_states", np.array(states))
        np.savez(f"{TRAINING_DATA_PATH}/{game.me.halite_amount}_{game.my_id}_{timestamp}_actions", np.array(action_indices))
        np.savez(f"{TRAINING_DATA_PATH}/{game.me.halite_amount}_{game.my_id}_{timestamp}_rewards", np.array(rewards))
        np.savez(f"{TRAINING_DATA_PATH}/{game.me.halite_amount}_{game.my_id}_{timestamp}_dones", np.array(dones))

    game.end_turn(command_queue)
