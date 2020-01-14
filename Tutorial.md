# Deep Reinforcement Learning for Halite III

[Halite III](http://www.halite.io "Halite III") is a game developed by two sigma.
Its a competition where you implement the behavior of bots to collect resources.

The key rules of the game are:
* You start with 5000 resource in your "bank" and can build ships in your main base. 
* A ship costs 1000 resource and can either move up, down, left, right, collect resource or build a depot.
* Ships can move around the playfield and collect resource that they need to drop off at either your main base or a depot
* Moving around costs 10% of the resource on a given field
* Collecting resource takes one turn and collects 25% of the resource on the given field
* Building another depot (to reduce travel times) costs 4000 resource
* Winner is whoever has most resource in the bank at the end of all turns

As you can see there is quite some complexity to the game. For this tutorial(series) I want to focus on building a reinforcement learning framework that focuses on moving ships, and collecting resources, maybe even decide when and how many ships to build but the first part is already complex enough.

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

## The baseline: "random" bot

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

* `Q_t` is the reward that we expect from the action we took in the current step t
* `l` is the learning rate (e.g. 0.1)
* `r` is the reward observed at time t
* `d` is to discount future reward, since we are not sure how much the current move contributed to the future reward (e.g. 0.95)
* `Q_t+1` is the maximum expected reward of the next state (according to our table)

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

We go through all stored states, actions, rewards etc. and if we are in a final state we set the actual reward of the action in the current state to the reward. If we are not in a final state we apply above formula to estimate the reward tht we will make given this move. 
The last line is adding some logging so we can get a glimpse of how it will look like when our table is learning.

Last we need to write the table to disk to store the state:

```python
        logging.info(f"Writing to {TABLE} - {len(table)} - {len(states)}")
        with open(TABLE, 'w') as f:
            f.write(json.dumps(table))

    game.end_turn(command_queue)
```
## Some more stuff for easier debugging and some results

To run this you can use the `run_game.sh` with something like this:

```sh
#!/bin/sh

$TABLE=$1
$EPSILON=$2

./halite --seed 123 --replay-directory replays/ --turn-limit 10 --no-logs -vvv --width 16 --height 16 "python3 QTableBot.py $EPSILON $TABLE" "python3 QTableBot.py 0.5 table_right"
```

you can call it with
```
./run_game.sh myTable 0.0
```
which will run a game using no random moves and just moving from our table.

Unless you specified the `--no-replay` option you can find a replay in the `replays` folder (a file ending on `*.hlt`) and watch it on the [halite](https://2018.halite.io/watch-games) website by just dragging it there.

Also you should be able to see that halite engine produces a bot-0.log and bot-1.log.
Inside you should be able to see the logging we implemented above at the end of each file like

```
INFO:root:Initializing myTable
INFO:root:Successfully1 created bot! My Player ID is 0.
INFO:root:=============== TURN 001 ================
INFO:root:=============== TURN 002 ================
INFO:root:=============== TURN 003 ================
INFO:root:=============== TURN 004 ================
INFO:root:=============== TURN 005 ================
INFO:root:=============== TURN 006 ================
INFO:root:=============== TURN 007 ================
INFO:root:=============== TURN 008 ================
INFO:root:=============== TURN 009 ================
INFO:root:=============== TURN 010 ================
INFO:root:c260d67d114e675abaa37c912bdf21787396828c704008d228e88efe | (-1, 0) | -1 | False | [0.4, 0.57, 0.81, -0.06, 0.46, 1]
INFO:root:3db150a61f98c9874a2a17cae4de1cd25bdfbc301e9f28f6a8aeaa3a | (0, 0) | -1 | False | [0.68, 0.22, 0.21, 0.43, 0.03, 1]
INFO:root:211d0d516807bcdc7d01ee2bedde52f04b367b106ba8b992a1539a00 | (0, 0) | -1 | False | [0.17, 0.86, 0.62, 0.3, -0.05, 1]
INFO:root:fb437b9e1ad15afb0ba219ce18d366332e558743f351913985ae4200 | (-1, 0) | -1 | False | [0.72, 0.03, 0.73, -0.02, 0.51, 1]
INFO:root:7ee0d73b982b3fa50ab35788bb105242f309060f23211d8907e83e6f | (0, -1) | -1 | False | [-0.03, 0.24, 0.2, 0.07, 0.33, 1]
INFO:root:ada7d3f9d171da0852c0600ca4d391e5274cb42bcfccd21981ef82ea | (0, -1) | -1 | False | [-0.03, 0.66, 0.62, 0.03, 0.74, 1]
INFO:root:98aeaafdbfb462fdbca72f8074096ae175a95cfce8d169452c60ed3d | (0, -1) | -1 | False | [-0.01, 0.79, 0.05, 0.51, 0.13, 1]
INFO:root:40afd21f83d94948ad04976c64e3a5218d0e8b761df5ebdbd6730181 | (0, 1) | -1 | False | [0.05, 0.43, 0.02, 0.13, 0.51, 1]
INFO:root:Writing to table_left - 9 - 9
```

We can see the hashes of each state, which move was chosen in terms of `(dx, dy)` and whether its a terminal state and then the array containing the expected future reward, where in our case the negative entries are already updated values according to our learning formula. 
You may notice the last number 1 which is some additional debugging that I implemened to see how often we visted a given state

if we rerun the same command with no exploration we can see

```
INFO:root:c260d67d114e675abaa37c912bdf21787396828c704008d228e88efe | (0, 1) | -1 | False | [0.4, 0.57, -0.11, -0.06, 0.46, 2]
INFO:root:0914ee6d8bbb4b23b58e9541c49c824406d7111877e75aa037b98429 | (0, 1) | -1 | False | [0.66, 0.82, -0.36, 0.67, 0.32, 1]
INFO:root:c371d77cf617a467d1f57157facb3c4e5abb7912c8a2424d8d1409a6 | (1, 0) | -1 | False | [0.21, -0.13, 0.31, 0.29, 0.07, 1]
INFO:root:9ceea48b8bfa5195a19992bc818612b59ccb3bd39c9034acf33b955a | (0, 0) | -1 | False | [0.18, 0.06, 0.66, 0.54, -0.45, 1]
INFO:root:0b0eb2b022140fc937b69963a1c4834d8cb350dbc52551605e8f33f6 | (0, 0) | -1 | False | [0.39, 0.24, 0.14, 0.17, -0.03, 1]
INFO:root:04c3cbf50f22b78fb5b2c2d1939cbd802d2df327d01ea2f9ca8270ae | (0, 0) | -1 | False | [0.08, 0.76, 0.83, 0.76, -0.17, 1]
INFO:root:34af96a04d8e09731226dc264ba5dc7ba0c6c3e2183f4617387ec4ca | (-1, 0) | -1 | False | [0.63, 0.0, 0.44, 0.03, 0.59, 1]
INFO:root:a3a5965b7a6dc82a977641952f4e7f2cc5418c54e3e0dd5f1b238d3b | (1, 0) | -1 | False | [0.33, 0.05, 0.41, 0.27, 0.39, 1]
INFO:root:Writing to table_left - 17 - 9
```

This time we visited the first state (beginning state) twice and we took different moves than before. This is because in the previous run we updated all our moves according to the reward we got (which was -1) so some moves now have negative values.
While this may seem like we could just run it like that and we would eventually find some good move patterns it is better to start with true random moves.

If we run it like we implemented it though we will find that it will take incredibly long for our table to learn anything though this is because there is just so many states in our game. To actually show some interesting output we should take care of a couple of things:

* Replace the oponent bot with a bot that does nothing, since when our oponent moves around and collects halite it alters the state of the game. So even if our bot makes the exact same moves it will look like a new state

If we do this we can add another bash script `batch_run_qtable.sh` to simulate a couple hundred games:

```
#!/bin/sh
TABLE=$1
EPSILON=$2

for i in {1..500000}
do
  ./halite --seed 123 --turn-limit 10 --replay-directory replays/ --no-timeout --no-replay --no-logs -v --width 16 --height 16 "python3 QTableBot.py $2 $1" "python3 DoNothingBot.py"
  echo $i
  ls -lh ./$1
done
```

Now we can just run this and kill it with ctrl-c if we think we have had enough.
So I ran it for ~500 games with 1.0 epsilon and then one `run_game.sh` with 0.0 epsilon to see the "best" path it know so far.

```
INFO:root:9b007...872e6 | (0, 1) | -1 | False | [41.5, 20.3, 98.92, 58.83, 92.9, 495]
INFO:root:b55eb...ff67d | (1, 0) | -1 | False | [104.96, 105.2, 104.91, 105.02, 104.55, 99]
INFO:root:f61bf...06ed8 | (0, 1) | -1 | False | [40, 21.47, 111.8, -0.05, 80.7, 99]
INFO:root:8b161...3375c | (-1, 0) | -1 | False | [106.87, 106.87, 115.62, 118.85, 118.64, 32]
INFO:root:2b1ff...7280f | (0, -1) | -1 | False | [126.29, -0.88, -1.05, -0.93, -0.2, 32]
INFO:root:0048c...d4e0b | (0, -1) | 134 | True | [134, 0.62, -0.36, -0.2, -0.13, 8]
INFO:root:0c6e0...c2a46 | (0, 0) | -1 | False | [0.0, 0.15, 0.02, 0.15, 0.02, 2]
INFO:root:0c6e0...c2a46 | (0, 0) | -1 | False | [0.0, 0.15, 0.02, 0.15, -0.77, 3]
INFO:root:Writing to myTable10Turns - 1114 - 9
```

Next we run it with EPSILON set to 0.5 so we make every second move random, for about 200 episodes


```
INFO:root:9b007...872e6 | (0, 1) | -1 | False | [52.67, 46.43, 187.3, 58.9, 176.84, 714]
INFO:root:b55eb...ff67d | (0, 1) | -1 | False | [197.35, 198.16, 198.21, 198.2, 198.21, 234]
INFO:root:f61bf...06ed8 | (0, 1) | -1 | False | [40, 29.35, 209.69, 11.57, 106.34, 234]
INFO:root:8b161...3375c | (0, -1) | -1 | False | [221.78, 173.23, 211.46, 221.77, 221.29, 117]
INFO:root:2b1ff...7280f | (0, 0) | -1 | False | [144.29, -1.38, -1.14, -1.08, 234.5, 117]
INFO:root:abd98...d1b95 | (0, -1) | -1 | False | [247.9, 0.04, -0.06, -0.61, -0.04, 20]
INFO:root:328a0...0e63d | (0, -1) | 262 | True | [262, -0.18, -0.17, -0.13, 0.6, 13]
INFO:root:9e266...da1ac | (0, 1) | -1 | False | [-0.19, -0.11, -0.1, -0.1, -0.23, 9]
INFO:root:Writing to myTable10Turns - 1398 - 9
```

then we run it another 100 episodes with EPSILON 0.1

```
INFO:root:9b007...872e6 | (0, 1) | -1 | False | [52.67, 49.2, 209.05, 58.9, 197.58, 856]
INFO:root:b55eb...ff67d | (0, 1) | -1 | False | [218.73, 218.82, 221.11, 198.21, 218.82, 367]
INFO:root:f61bf...06ed8 | (0, 1) | -1 | False | [40, 34.76, 233.8, 41.93, 106.34, 367]
INFO:root:8b161...3375c | (0, 1) | -1 | False | [244.65, 216.92, 247.16, 247.13, 244.62, 243]
INFO:root:2b1ff...7280f | (0, 0) | -1 | False | [144.29, -1.41, -1.14, 93.91, 261.22, 243]
INFO:root:abd98...d1b95 | (0, -1) | -1 | False | [276.02, -0.86, -0.06, -0.64, -0.24, 139]
INFO:root:328a0...0e63d | (0, 0) | -1 | False | [262, -0.79, -0.91, -0.16, 291.6, 125]
INFO:root:12a62...a5aa3 | (0, -1) | 308 | True | [308, -0.01, -0.2, -0.18, -0.07, 59]
INFO:root:Writing to myTable10Turns - 1441 - 9
```

We can see how it converged from 143 to 262 to finally 308 resource collected, looking at the movement pattern it seems like the bot only collected resource twice which seems strange. By comparing this with the replay I found that if a bot cannot move because it cannot pay the fuel cost to move it will default to standing still, which in our case attributes the reward slightly wrong. the second and fourth move in the list were actually stand stills.
The bot moved down twice and up twice. (if you just add the tuples you would otherwise not end up at the orignal place when the bot says its done)

Fixing these issues and letting it run for ~20k random walks we find the perfect solution.

```
INFO:root:314a1...7a9a0 | (-1, 0) | -1 | False | [135.73, 129.44, 187.3, 283.08, -1, 19148]
INFO:root:e2d69...b8d1f | (0, 0) | -1 | False | [0.91, 0.81, 0.22, 0.17, 299.03, 4461]
INFO:root:cbc96...b0c1a | (-1, 0) | -1 | False | [91.4, 25, 91.37, 315.82, 251.6, 4461]
INFO:root:c75fa...d1d96 | (0, 0) | -1 | False | [0.33, 0.23, 0.92, 0.52, 333.5, 1492]
INFO:root:ab6f3...50ed6 | (0, 0) | -1 | False | [96.6, 165.21, 100.04, 68.31, 352.1, 1492]
INFO:root:41f40...2278b | (0, 0) | -1 | False | [-2.85, 296.77, -2.85, -2.85, 371.68, 757]
INFO:root:f76dd...049a9 | (1, 0) | -1 | False | [-1.95, 392.3, -1.95, -1.95, -1.95, 558]
INFO:root:42f66...674a1 | (1, 0) | 414 | True | [-1, 414, -1, -1, -1, 438]
INFO:root:Writing to myTableConditionedOnTurnS - 10859 - 9
```
![QTableBot](https://user-images.githubusercontent.com/1778723/72352937-7a90af80-36e3-11ea-82d3-cb3664c7d08e.gif)

It is also interesting to note that while the initial state was seen 19148 times we only have a total of 10859 keys, depending on the size of our state space this can increase drastically.
If we take the ratio it kind of shows us that on average every state was vistited 1.8 times. Of course the true distribution of visits is very left skewed with the first state being always vistied and the next ones already only ~4500 times as we can see above.
So while this already helped find a couple of issues with our reward function and quite some bugs (that I didn't describe here but stuff didn't quite work out of the box)
the result is nice but also a little bit: so what? we could have probably written a breath/depth first search or some other heuristic and found that solution much easier than implementing such a complicated solution. We are only doing the first 10 steps of the game and not considering anything else.
And while this is true for the example that I showed here it all depends on how we encode our state.
Can we somehow encode the state of the game in a different way that helps us generalize better being able to reuse the state for many ships and many situations?
A couple of ideas that pop to my mind are:

* Bucket/binning of the resoruces in each field, since each change in resource on a field changes our state, we could round/bucket the resource amount to the nearest 5,10 or 50, so collecting resource on a field would not always change its value
* Center the state "map" around the ship itself instead of making it a global state of the map, so each ship would see itself as the center of the universe
* Have a smaller area around the ship where we have full resolution of the game map and aggregate everything further away, e.g. on a 16x16 map we could have a ship that sees a 4x4 surrounding
* Use neural networks and just generalize over all of that and don't bother with the "rigidness" of the problem but open up a whole other can of worms.

The latter is what we will do next :)


