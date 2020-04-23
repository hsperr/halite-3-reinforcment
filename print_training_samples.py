import numpy as np
import tensorflow as tf

import sys

MODEL_PATH = "experiments/positive_reward_for_moving_around-741e1753b71b2b6b2879a507a69a00f8933bca84317a40e04a011d77/model"
model = tf.keras.models.load_model(MODEL_PATH)

FILE = sys.argv[1]

BASE_PATH = FILE.replace("_states.npz", '')

states = np.load(f"{BASE_PATH}_states.npz", allow_pickle=True)["arr_0"]
actions = np.load(f"{BASE_PATH}_actions.npz", allow_pickle=True)["arr_0"]
rewards = np.load(f"{BASE_PATH}_rewards.npz", allow_pickle=True)["arr_0"]
dones = np.load(f"{BASE_PATH}_dones.npz", allow_pickle=True)["arr_0"]

current_qs = model.predict(states)
train_y = []
for state, current_q, future_q, action_index, reward, done in zip(states, current_qs, current_qs[1:], actions, rewards, dones):

    new_q_value = reward
    if not done:
        new_q_value = reward + 0.95 * future_q[np.argmax(future_q)]


    new_q = np.array([x for x in current_q])
    new_q[action_index] = new_q_value

    train_y.append(new_q)


print("Sum rewards:", np.sum(rewards))
print("Total positive rewards:", np.sum(rewards[rewards>0]))
print("Total negative rewards:", np.sum(rewards[rewards<0]))
print("Samples:", len(states), len(actions), len(rewards), len(dones))

print("****************** Done States")
for pred, action, r, d, y in zip(model.predict(states), actions, rewards, dones, train_y):
    if d:
        print(r, d, pred, action, np.argmax(pred))

cnt = 0
print("****************** First 10")
correct = 0
for pred, action, r, d, y in zip(model.predict(states), actions, rewards, dones, train_y):
    if cnt <10:
        print(r, d, pred, action, np.argmax(pred))

    correct+=(action==np.argmax(pred))
    cnt+=1

print(correct, cnt, correct/cnt)
