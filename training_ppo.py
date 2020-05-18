import numpy as np
import tqdm
import os

import tensorflow as tf
# from keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95

def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae

        # print(f"i={i}, mask={masks[i]}, reward={rewards[i]}, value={values[i]}, value+1={values[i+1]}, delta={delta}, gae={gae}, ret={gae+values[i]}")
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

def ppo_loss_np(y_true, y_pred, oldpolicy_probs, advantages):
    # print('y_true:', y_true)
    # print('y_pred/newpol:', y_pred)
    # print('old_policy:', oldpolicy_probs)
    newpolicy_probs = y_pred

    ratio = np.exp(np.log(newpolicy_probs + 1e-10) - np.log(oldpolicy_probs + 1e-10))
    # print('ratio: ', ratio)
    p1 = ratio * advantages
    # print('advantages: ', advantages.flatten())
    # print('p1: ', p1)
    p2 = np.clip(ratio, a_min=1 - clipping_val, a_max=1 + clipping_val) * advantages
    # print('p2: ', p1)
    actor_loss = -np.mean(np.minimum(p1, p2))
    print('actor_loss: ', actor_loss)
    # critic_loss = np.mean(np.square(rewards - values))
    # print('rewards: ', rewards.flatten())
    # print('values: ', values.flatten())
    # print('critic_loss: ', critic_loss)

    # term_a = critic_discount * critic_loss
    # print('term_a: ', term_a)
    # term_b_2 = np.log(newpolicy_probs + 1e-10)
    # print('term_b_2: ', term_b_2)
    # term_b = entropy_beta * np.mean(-(newpolicy_probs * term_b_2))
    # print('term_b: ', term_b)
    # total_loss = term_a + actor_loss - term_b
    # print('total_loss: ', total_loss)
    return actor_loss


def ppo_loss_print(oldpolicy_probs, advantages, rewards, values):
    import sys
    def loss(y_true, y_pred):
        print('y_true:', y_true)
        print('y_pred/newpol:', y_pred)
        print('old_policy:', oldpolicy_probs)
        newpolicy_probs = y_pred

        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        print('ratio: ', ratio)
        print('advantages: ', advantages)
        p1 = ratio * advantages
        print('p1: ', p1)
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        print('p2: ', p1)
        actor_loss = -K.mean(K.minimum(p1, p2))
        print('actor_loss: ', actor_loss)
        critic_loss = K.mean(K.square(rewards - values))
        print('rewards: ', rewards)
        print('values: ', values)
        print('critic_loss: ', critic_loss)

        term_a = critic_discount * critic_loss
        print('term_a: ', term_a)
        term_b_2 = K.log(newpolicy_probs + 1e-10)
        print('term_b_2: ', term_b_2)
        term_b = entropy_beta * K.mean(-(newpolicy_probs * term_b_2))
        print('term_b: ', term_b)
        total_loss = term_a + actor_loss - term_b
        print('total_loss: ', total_loss)
        return total_loss

    return loss


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss \
                     - entropy_beta * K.mean( -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return actor_loss

    return loss


def get_model_actor(input_dims, output_dims):
    state_input = Input(shape=(input_dims))
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    x1 = Conv2D(32, (1, 1), strides=1, activation='linear', padding="same", data_format='channels_last')(state_input)
    x1 = Activation("relu")(x1)
    layer1_out = BatchNormalization()(x1)

    x1 = Conv2D(32, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(layer1_out)
    x1 = Activation("relu")(x1)
    x1 = BatchNormalization()(x1)

    prev = x1
    for i in range(1):
        x1 = Conv2D(32, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(prev)
        x1 = Activation("relu")(x1)
        x1 = BatchNormalization()(x1)
        prev = x1

    x = Flatten()(x1)
    x = Dense(128, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)], experimental_run_tf_function=False)

    model.summary()
    return model

def get_model_critic(input_dims):
    state_input = Input(shape=(input_dims))
    x1 = Conv2D(32, (1, 1), strides=1, activation='linear', padding="same", data_format='channels_last')(state_input)
    x1 = Activation("relu")(x1)
    layer1_out = BatchNormalization()(x1)

    x1 = Conv2D(32, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(layer1_out)
    x1 = Activation("relu")(x1)
    x1 = BatchNormalization()(x1)

    prev = x1
    for i in range(1):
        x1 = Conv2D(32, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(prev)
        x1 = Activation("relu")(x1)
        x1 = BatchNormalization()(x1)
        prev = x1

    x = Flatten()(x1)
    x = Dense(128, activation='relu', name='fc1')(x)
    out_actions = Dense(1, activation='linear')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', experimental_run_tf_function=False)
    model.summary()
    return model

state_dims = (8,8,6)
n_actions = 5

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

# tensor_board = TensorBoard(log_dir='./logs')

model_actor = get_model_actor(input_dims=state_dims, output_dims=n_actions)
model_critic = get_model_critic(input_dims=state_dims)

ppo_steps = 10
target_reached = False
best_reward = 0
iters = 0
max_iters = 50

def load_samples(path):
    import glob
    total_reward = 0
    states, actions, rewards, dones = [], [], [], []
    files = [x for x in glob.glob(path+"/*_states.npz")]
    print(path, len(files))
    for file in tqdm.tqdm(files):
        total_reward += int(file.split('/')[-1].split('_')[0])
        try:
            BASE_PATH = file.replace("_states.npz", '')
            states.extend(np.load(f"{BASE_PATH}_states.npz", allow_pickle=True)["arr_0"])
            actions.extend(np.load(f"{BASE_PATH}_actions.npz", allow_pickle=True)["arr_0"])
            rewards.extend(np.load(f"{BASE_PATH}_rewards.npz", allow_pickle=True)["arr_0"])
            dones.extend(np.load(f"{BASE_PATH}_dones.npz", allow_pickle=True)["arr_0"])
        except Exception as e:
            print(file, "error:", e)
    print(f"Loaded num_states={len(states)}, total_harvest={total_reward}, avg_harvest={round(total_reward/len(states), 2)}")
    print(f"Rewards earned={np.sum(rewards)}, avg={np.mean(rewards)}")
    return states, actions, rewards, dones

if not os.path.exists("./ppo"):
    os.mkdir("./ppo")
if not os.path.exists("./ppo/data"):
    os.mkdir("./ppo/data")
if not os.path.exists("./ppo/models"):
    os.mkdir("./ppo/models")

current_actor = './ppo/models/model_actor_{}_{}.hdf5'.format(iters, "START")
current_critic = './ppo/models/model_critic_{}_{}.hdf5'.format(iters, "START")

model_actor.save(current_actor)
model_critic.save(current_critic)

print("Initialized everything")


def play(data_path, deterministic=0):
    command = './halite ' \
          '--replay-directory replays/ ' \
          '--no-timeout ' \
          '--turn-limit 100 ' \
          '--no-logs -v ' \
          '--no-replay ' \
          f'--width {FIELD_SIZE} ' \
          f'--height {FIELD_SIZE} ' \
          f'"python3 PPOBot.py {current_actor} {data_path} {deterministic}" ' \
          f'"python3 PPOBot.py {current_actor} {data_path} {deterministic}" '

    print(command)
    os.system(command)


while not target_reached and iters < max_iters:

    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = []
    actions_onehot = []
    state_input = None

    FIELD_SIZE = 8
    DATA_PATH = f'ppo/data/{iters}'

    for x in tqdm.tqdm(range(ppo_steps)):
        play(data_path=DATA_PATH, deterministic=0)

    play(data_path=DATA_PATH, deterministic=1)

    states, actions, rewards, dones = load_samples(DATA_PATH)
    avg_reward = np.mean(rewards)

    values = model_critic.predict([np.array(states)], steps=1).flatten()
    dummy_ns = np.zeros((len(states), 1, n_actions))
    dummy_1s = np.zeros((len(states), 1, 1))

    actions_probs = model_actor.predict([np.array(states), dummy_ns, dummy_1s, dummy_1s, dummy_1s]).reshape(len(states), 1, 5)
    values = np.append(values, values[-1:])
    masks = [not x for x in dones]

    returns, advantages = get_advantages(values, masks, rewards)
    advantages = advantages.reshape(len(states), 1, 1)
    rewards = np.array(rewards).reshape(len(states), 1, 1)

    actions = np.array(actions)
    b = np.zeros((actions.size, actions.max() + 1))
    b[np.arange(actions.size), actions] = 1
    actions_onehot = b

    # for cnt, (s, a, d, rew, ret, adv, ap, v) in enumerate(zip(states, actions, dones, rewards, returns, advantages, actions_probs, values)):
    #     print(f"idx={cnt}, action={a}, done={d}, reward={rew[0][0]}, return={ret}, advantage={adv}, value={v}, prob={ap}")
    #     input()
    #
    #
    # input()


    # ppo_loss_np(actions_onehot, actions_probs, actions_probs, advantages, np.array(returns), values[:-1])

    actor_loss = model_actor.fit(
        [states, actions_probs, advantages, rewards, values[:-1].reshape(len(states), 1, 1)],
        actions_onehot,
        batch_size=198,
        verbose=True,
        shuffle=True,
        epochs=4
    )

    critic_loss = model_critic.fit(
        [states],
        [np.reshape(returns, newshape=(-1, 1))],
        batch_size=198,
        shuffle=True,
        verbose=True,
        epochs=4
    )

    current_actor = './ppo/models/model_actor_current.hdf5'
    model_actor.save(current_actor)

    TEST_PATH = DATA_PATH+"/test"
    if not os.path.exists(TEST_PATH):
        os.mkdir(TEST_PATH)
    play(TEST_PATH, 1)
    play(TEST_PATH, 1)
    play(TEST_PATH, 1)

    states, actions, rewards, dones = load_samples(TEST_PATH)
    avg_reward = np.mean(rewards)

    print(f'total test reward={avg_reward}, current best={best_reward}')
    if avg_reward > best_reward:
        current_actor = './ppo/models/model_actor_{}_{}.hdf5'.format(iters, avg_reward)
        model_actor.save(current_actor)
        model_critic.save('./ppo/models/model_critic_{}_{}.hdf5'.format(iters, avg_reward))
        best_reward = avg_reward
    iters += 1

