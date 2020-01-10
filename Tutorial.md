# Deep Reinforcement Learning for Halite III

[Halite III](http://www.halite.io "Halite III") is a game developed by two sigma. There was a competition in 2018 trying to build a bot that could collect the most amount of halite (resource).

The game consists of the key elements of:
* Moving ships around and using them to collect a resource
* Deciding whether to build more ships or have any ship convert into another base to shorten travel distances
* Having more halite in your bank at the end of the game than your opponent. So building ships etc is a tradeoff between having more resource in your bank or collecting faster

Checkout the page of the competiton above for more details about the rules.

Back when the competiton was on I first build a rule based bot which if I remember correctly eneded up in the top 200. Towards the end of the competition I got inspired by a YouTuber called [sentdex](https://www.youtube.com/sentdex) who got me into the competition in the first place.

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
Also this will be one of the rather few examples that does not use `Gym` or some other library that wraps the complexity of the game behind something that returns a nice state and reward and action representation. So far not many tutorials talk about it but I find making decisions on how to structure these things are not at all trivial and just implenenting a DQN that can learn cartpole is in my opinion good for learning the general algorithms but leaves me with very little knowledge on how to apply Reinforcement Learning in the real world

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

...

TABLE = sys.argv[1]
if os.path.exists(TABLE):
    with open(TABLE, "r") as f:
        table = json.loads(f.read())
else:
    table = {}

while True:

TODO
...
```

## Modelling Reward

Now potentially we have a table that can store the states and give us randomly initiallized expected rewards.
In order to train this table we should think a little bit about rewards in this game. 
There is a bunch of ways on how to model this but I found the simplest of them so far works quite OK.

If you want to dive into more detail about reward modelling and the research here google it a little bit.

The simplest way of defining rewards would be to have three distinct cases.

1. The ship is bringing resource to a base hence increasing our bank (reward = amount of resource banked)
2. The ship is crashing with another ship (reward = -1000, this is the cost of a ship, we could increase this)
3. The ship is doing anything else (reward = -1)

An episode is considered to be done when a ship returns resources to the base and I see the whole game as a sequence of multiple episodes for a ship.
Or however often it goes out to collect something.

TODO write more about different types of reward tried

## Updating our Tables and finally running experiments



```

```
## Getting your hands dirty

What we need to do in order to implement Q-Tables is, we need
