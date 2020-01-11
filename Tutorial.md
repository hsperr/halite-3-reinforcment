# Deep Reinforcement Learning for Halite III

[Halite III](http://www.halite.io "Halite III") is a game developed by two sigma.
There was a competition in 2018 trying to build a bot that could collect the most amount of halite (resource).

The game consists of the key elements of:
* Moving ships around and using them to collect a resource
* Deciding whether to build more ships or have any ship convert into another base to shorten travel distances
* Having more halite in your bank at the end of the game than your opponent. So building ships etc is a tradeoff between having more resource in your bank or collecting faster

Checkout the page of the competiton above for more details about the rules.

Back when the competiton was on I first build a rule based bot which if I remember correctly eneded up in the top 200. Towards the end of the competition I got inspired by a YouTuber called [sentdex](https://www.youtube.com/sentdex) who got me into the competition in the first place.
Also a really good reference on how to encode the game into a neural network is the writeup of [Joshua Staker](https://stakernotes.com/).

He tried to apply neural networks aswell starting from a random bot which would learn by playing itself over and over again.
I am not sure his efforts lead to a final version which did things but it was really hard to model anything since there is a lot of noise in just random games. Things like prioritizing big chunks of resource vs small ones and venturing further for bigger reward are hard to model since a random walk is most likely never going to return to the base again to hand it off.

I then started to download the replays of the top players and tried to learn them using neural networks and various approaches but it never lead to anything. I also tried some Deep Q-Learning tutorials back then and got the ships to not run into each other all the time and collect a small amount of halite. This was a lot of fudging everything together and somehow changing the reward function mid training depending on which issues the current bot had (e.g. just not crashing ships is good).

Anyway now almost a year later, while I was waiting for Halite IV to be honest I decided to give the reinforcement learning a try again.

This time I took the approach to dumb the game down until it is really small parts that I try to model to see if it works at all or not.

For this I tried...
* making the game map smaller.
* only running the game for 8 or 50 turns instead of the usual 400, since if we can't find the best solution for a 8 turn game we will never model the 400 turn solution.
* only building one ship instead of multiples so crashing is not an issue.
* smaller and shallow networks as opposed to big deep networks or complicated architectures eventually even falling back to Q-Tables and trying to store all states of a small map with only 8 turns and one ship.

This will be the written step by step tutorial including results that I think are worth sharing.
Also this will be one of the rather few examples that does not use `Gym` or some other library that wraps the complexity of the game behind something that returns a nice state and reward and action representation. So far not many tutorials talk about it but I find making decisions on how to structure these things are not at all trivial and just implenenting a DQN that can learn cartpole is in my opinion good for learning the general algorithms but leaves me with very little knowledge on how to apply Reinforcement Learning in the real world.

## The basic "random" bot

This is the version of the bot that comes with the starter pack. We will later modify this bot to make sure we store the information we need to train our reinforcement learning algorithm.

I will focus on the game loop since the rest for now is just setting up and initializing the game engine.

Here is the code (it is well commented so I won't go through it line by line):

```Python

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    for ship in me.get_ships():
        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.
        if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
            command_queue.append(
                ship.move(
                    random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])))
        else:
            command_queue.append(ship.stay_still())

    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

```

So this bot moves random if the ship is reasonably full or collects halite if it is not.
It also spawns new ships in the first 200 turns.

We will use this bot as a baseline.

# Q-Tables

I did not first try Q-Tables but after playing around a lot with Deep Q-learning and nothing working I decided to make the problem simpler and simpler until I can get an Idea of whats happening.
This then helped me actually find a couple of bugs in my code which after fixing them let me progress with my neural networks so I want to show this even though we will not be able to do a lot with Q-Tables and this game.

Generally Q-Tables is just a way to store each possible state of the game and the expected rewards for every possible action from that state.
Inititally you just randomly initialize the table and take random actions checking which rewards you get updating your table and your actions along the way.

If you want to get a good idea of what Q-Tables are just google around there is plenty of pretty good medium posts and other resources, I just want to list a couple

* Again a shoutout to sentdex for his video [Q Learning Intro/Table - Reinforcement Learning p.1](https://www.youtube.com/watch?v=yMk_XtIEzH8)
* Also Arthur Juliani for his series on [Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

Again thers is plenty other really good resources out there.

## Representing State

Since I first did neural networks and then q-tables I reused the features from the neural networks to represent the state in my q-tables aswell.
I modelled it as a `6 x size x size`  where `size` is the size of the playfield e.g. 32 (then it would be a 6x32x32 matrix)

My 6 features are:

1. The amount of resource on the field
2. Whether there is a own ship on this field or not (modelled as a proportion of how much resource the ship is carrying)
3. Whether there is a enemy ship on this field (also modelled as a proportion)
4. Whether there is a own base on this field (boolean)
5. Whether there is a enemy base on this field (boolean)
6. One hot encoded matrix of the ship that is currently to decide the move

Here is the code

```Python
def create_state_array(game_map, ship):
    state = np.zeros((6, game_map.width, game_map.height))
    for cell in game_map.cells:
        state[0, cell.position.x, cell.position.y] = cell.halite_amount/1000.0

        state[1, cell.position.x, cell.position.y] = (cell.ship.halite_amount+1)/1001 if cell.is_occupied and cell.ship.owner == ship.owner else 0
        state[2, cell.position.x, cell.position.y] = (cell.ship.halite_amount+1)/1001 if cell.is_occupied and not cell.ship.owner == ship.owner else 0

        state[3, cell.position.x, cell.position.y] = 1 if cell.has_structure and cell.structure.owner == ship.owner else 0
        state[4, cell.position.x, cell.position.y] = 1 if cell.has_structure and not cell.structure.owner == ship.owner else 0

    state[5, ship.position.x, ship.position.y] = 1
    return state
```

Later we will feed this matrix into the neural network but for now we will use this to represent the state for our Q-Table.
I felt the easiest way to build such a table would be a dictionart of the above features to the array containing the expected reward for each action.
Since storing all the information above would bloat up the table significantly I though I can probably get around it by hashing the information above.

In our case, we also don't want to enumerate each possible state and initialize the table so we will do a lazy initialization. 
This means if we try to lookup a key in our table that is not there we will just initialize a random array instead.

```Python
import os
import hashlib
import sys

EPSILON = float(sys.argv[1])
TABLE = sys.argv[2]

if os.path.exists(TABLE):
    logging.info(f"Loading {TABLE}")
    with open(TABLE, 'rb') as f:
        table = json.loads(f.read())
else:
    logging.info(f"Initializing {TABLE}")
    table = {}

...
```

## Making a move

To decide which move we want to make, we now depending on the EPSILON parameter specified either chose a random move or 
we lookup the most promising move according to our table. If the table does not contain the current game state we will
insert an array with random numbers between 0 and 1.

Lets go through the code needed bit by bit:

```python
POSSIBLE_MOVES = [Direction.North, Direction.East, Direction.South, Direction.West, Direction.Still]

def state2key(state):
    return hashlib.sha224(f"{state.tostring()}".encode("UTF-8")).hexdigest() 

def lookup_table(table, state):
    return table.setdefault(state2key(state), list(np.random.uniform(0, 1, len(POSSIBLE_MOVES))))
...
while True:
```

We use hashlib to create a stable hash between runs, in case you are wondering but since python 3 the normal `hash` function has a randomized seed
for security purposes. Hashing will help us to compress the multiple numpy arrays that we use to store our state into a smaller string representation.
Technically we could have collisions on our hashes but this is unlikely and I am happy to live with this risk.
I moved the hashing and the looking up of the table into functions since we move it a couple of times and replicating this code with the setdefault and such would be a bit ugly.

Next lets look at how we decide where to move:
```python
    for ship in me.get_ships():
        if random.random() < EPSILON:
            move = random.choice(POSSIBLE_MOVES)
        else:
            state = create_state_array(game_map, ship)
            current_q = lookup_table(table, state)
            move_index = np.argmax(current_q)
            move = POSSIBLE_MOVES[move_index]
        command_queue.append(ship.move(move))
```

We generate a random number, if its smaller than the chosen EPSILON, which is our exploration parameter then we chose a random move.
If the random number was greater or equal to our EPSILON then we create the current state, look it up in our table and take the one promising the maximum reward. 
In the current state without any training or learning if we did not have an EPSILON or set it to 0 we will generate one random walk (the first initialization is random) but then always walk the same way afterwards. 

## Training the table

Before we can start actually learning anything we need to start storing some data. 
So on top in the initialization part we add:

```python
states = []
action_indices = []
rewards = []
dones = []

shipid2shipandcell = {}
...
while True:
...
     for ship in me.get_ships():
            move = random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])
            state = create_state_array(game_map, ship)
            states.append(state)
            if random.random() < EPSILON:
                move = random.choice(POSSIBLE_MOVES)
            else:
                current_q = lookup_table(table, state)
                move_index = np.argmax(current_q)
                move = POSSIBLE_MOVES[move_index]

            action_indices.append(POSSIBLE_MOVES.index(move))
            shipid2shipandcell[ship.id] = (ship.halite_amount, game_map[ship.position].halite_amount)

            command_queue.append(ship.move(move))
...

```
The arrays are used to store the various things we need. People familiar with the openAI gym will recognize these.
Then we have to slightly adjust our move function again to make sture we store the state and chosen move during each iteration
The `shipid2shipandcell` variable is used to store the current ship halite amount and the halite of the current cell of this ship
Once we implenent our reward function we will see why we need this helper. You can ignore this for now.

### Modelling Reward

Now potentially we have a table that can store the states and give us randomly initialized expected rewards.
In order to train this table we should think a little bit about rewards in this game. 
There is a bunch of ways on how to model this but I found the simplest of them so far works quite OK.

If you want to dive into more detail about reward modelling and the research here google it a little bit.

The simplest way of defining rewards would be to have three distinct cases.

1. The ship is bringing resource to a base hence increasing our bank (reward = amount of resource banked)
2. The ship is crashing with another ship (reward = -1000, this is the cost of a ship, we could increase this)
3. The ship is doing anything else (reward = -1)

An episode is considered to be done when a ship returns resources to the base and I see the whole game as a sequence of multiple episodes for a ship.
Or however often it goes out to collect something.

A bit tricky bit is now that since we cannot observe the reward of the current move we need to calculate the reward for the previous move of the ship.
We do this by calling this reward function before we plan the moves of our ship, so in the first round it won't do anything.
As you might remember the `shipid2shipandcell` dictionary is filled when we make moves for our ships, so in the first round it will be empty.
Also we need a way to figure out whether our ship died last round, so we need to get all ids of current ships and check if a ship that we moved last round disappeared. The whole ordering of this may sound complicated but if you take a look at the full code here at [github](https://github.com/hsperr/halite-3-reinforcment/blob/b0804a9e45c318ab689b918b8d074f88fa240351/QTableBot.py) it should be ok.

```python
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
            dones.append(False)

    shipid2shipandcell = {}
```

So for each ship that we moved the round before we check whether it is still alive, if not we are done and the reward is -1000
If the ship is on our shipyard and it had some halite loaded in the previous turn. We use the stuff we stored to calculate 
how much halite we actually dropped off. 
This is not so straight forward since the ship may still have to pay fuel moving onto the home. We could compare the bank score before to the bank score now to see how much ended up on our account. If we were to play with more than one ship though this may not be accurate since two ships could drop something of at the same time, a ship could be built the same time or another ship may build a dropoff all causing the global counter to change.
In the current model we could simplify this since we know we only have one ship etc but I want to reuse this function later when we get to deep Q learning.

After we created the rewards we need to reset the `shipid2shipandcell` dictionary to make sure we clear our anything for not needed for next round (e.g. dead ships)

## Finally the training itself and some debug outputs

Now that we store all the information we need we can start actually updating our table.
the following code goes below the for loop for deciding the moves for each ship:

```python
    if len(my_ship_ids)<1:
        command_queue.append(me.shipyard.spawn())
```
In our Q-Table version we only build one ship for now, to reduce the state space as much as possible.
The formular for updating the Q-table is:

![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20Q_t%20%3D%20l*Q_t&plus;%281-l%29*%28r&plus;d*%5Carg%5C%21%5Cmax%20Q_%7Bt&plus;1%7D%29)

In code it looks like this:

```python
    if game.turn_number == constants.MAX_TURNS:
        for state, next_state, action_index, reward, done in zip(states, states[1:], action_indices, rewards, dones):
            current_q = lookup_table(table, state)
            future_q = lookup_table(table, next_state)

            if done:
                current_q[action_index] = reward
            else:
                current_q[action_index] = LEARNING_RATE * current_q[action_index] + (1 - LEARNING_RATE) * (reward + DISCOUNT * np.max(future_q))

            table[state2key(state)] = current_q

            logging.info(f"{state2key(state)} | {POSSIBLE_MOVES[action_index]} | {reward} | {done} | {[round(x, 2) for x in current_q]}")
```
This is the main training loop. Unfortunately Halite engine does not call us after the last round again to store away any state. 
It will just kill our process, so we need to have our own inetrnal decision on when to do that. So we check if the current round is the last.

We go through all stored states, actions, rewards etc. 


```python
        logging.info(f"Writing to {TABLE} - {len(table)} - {len(states)}")
        with open(TABLE, 'w') as f:
            f.write(json.dumps(table))

    game.end_turn(command_queue)
```
## Getting your hands dirty

What we need to do in order to implement Q-Tables is, we need
