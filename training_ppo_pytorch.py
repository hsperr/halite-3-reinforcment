
import numpy as np
import shutil
import tqdm
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader


class PPO(nn.Module):
    def __init__(self, num_outputs, hidden_size=128):
        super(PPO, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=32,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.actor = nn.Sequential(
            nn.Linear(32*8*8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.critic = nn.Sequential(
            nn.Linear(32*8*8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        base_val = self.base(x)
        batch_size = base_val.shape[0]
        base_val = base_val.view(batch_size, -1)

        value = self.critic(base_val)
        action_probs = F.softmax(self.actor(base_val), dim=-1)
        return action_probs, value.squeeze()

model = PPO(5).double()

FIELD_SIZE = 8
optimizer = optim.Adam(model.parameters(), lr=0.0001)

eps = np.finfo(np.float32).eps.item()

state_dims = (8,8,6)
n_actions = 5

ppo_steps = 10
test_steps = 25
epochs = 10

best_test_reward = 0
best_train_reward = 0
iters = 0
max_iters = 500

def load_samples(path):
    import glob
    total_reward = 0
    states, actions, rewards, dones = [], [], [], []
    files = [x for x in glob.glob(path+"/*_states.npz")]
    print(path, len(files))
    for file in files:
        total_reward += int(file.split('/')[-1].split('_')[0])-4000
        try:
            BASE_PATH = file.replace("_states.npz", '')
            states.extend(np.load(f"{BASE_PATH}_states.npz", allow_pickle=True)["arr_0"])
            actions.extend(np.load(f"{BASE_PATH}_actions.npz", allow_pickle=True)["arr_0"])
            rewards.extend(np.load(f"{BASE_PATH}_rewards.npz", allow_pickle=True)["arr_0"])
            dones.extend(np.load(f"{BASE_PATH}_dones.npz", allow_pickle=True)["arr_0"])
        except Exception as e:
            print(file, "error:", e)
    print(f"Loaded num_states={len(states)}, rewards={np.sum(rewards)}, total_harvest={total_reward}, avg_reward={np.mean(rewards)}")
    return states, actions, rewards, dones

if not os.path.exists("./ppo"):
    os.mkdir("./ppo")
if not os.path.exists("./ppo/data"):
    os.mkdir("./ppo/data")
if not os.path.exists("./ppo/models"):
    os.mkdir("./ppo/models")

current_actor = "./ppo/models/actor_critic_initial_pytorch.model"
torch.save(model, current_actor)

print("Initialized everything")
import subprocess
def play(data_path, deterministic=0, seed=None, replay=False):
    command = ['./halite' ,
          '--replay-directory replays/' ,
          '--no-timeout',
          '--turn-limit 100',
          '--no-logs',
          '-v',
          f'--width {FIELD_SIZE}',
          f'--height {FIELD_SIZE}'
    ]

    if not replay:
        command.append('--no-replay')

    if seed:
        command.append(f"--seed {seed}")

    command.append(f'"python3 PPOTorchBot.py {current_actor} {data_path} {deterministic}"')
    command.append(f'"python3 PPOTorchBot.py {current_actor} {data_path} {deterministic}"')

    subprocess.call(' '.join(command), shell=True)
    # os.system(command)



def finish_episode(states, actions, rewards, dones):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    cnt = len(rewards)
    for r, d in zip(rewards[::-1], dones[::-1]):
        R = r + 0.95 * R * (not d)
        cnt-=1
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # from collections import defaultdict
    # a2c = defaultdict(int)
    # a2r = defaultdict(int)
    # a2re = defaultdict(int)
    #
    # for a, r, ret in zip(actions, rewards, returns.detach().numpy()):
    #     a2c[a] += 1
    #     a2r[a] += r
    #     a2re[a] += ret
    #
    # for a, r in a2r.items():
    #     print(f"action={a}, return={r}, count={a2c[a]}, return={a2re[a]}")


    old_action_probs, old_values = model(states)
    old_dist = Categorical(old_action_probs)
    old_log_probs = old_dist.log_prob(actions)

    returns = calc_returns_actor_critic(rewards, dones)

    myData = TensorDataset(states, actions, returns, old_log_probs)
    myDataLoader = DataLoader(myData, batch_size=512, shuffle=True)

    for i in range(epochs):
        losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        for states, actions, returns, old_log_probs in myDataLoader:
            action_probs, values = model(states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()

            ratios = (log_probs - old_log_probs.detach()).exp()
            advantages = returns - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
            policy_loss = -torch.min(surr1, surr2)
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * dist_entropy
            loss = loss.mean()

            losses.append(loss.item())
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.item())
            entropy_losses.append(dist_entropy.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # take gradient step
        print(f"epoch={i}, samples={len(rewards)}, avg_loss={np.mean(losses)} -"
              f" policy={np.mean(policy_losses)} | value={np.mean(value_losses)} | entropy={np.mean(entropy_losses)}")



    # advantages = returns.sub(values.view(values.shape[0]))
    # policys_losses = -log_probs * advantages
    # values_losses = (returns - values).pow(2).mean()

    # advs = []
    #
    # for log_prob, value, R in zip(log_probs, values, returns):
    #     advantage = R - value.item()
    #     advs.append(advantage)
    #
    #     # calculate actor (policy) loss
    #     policy_losses.append(-log_prob * advantage)
    #
    #     # calculate critic (value) loss using L1 smooth loss
    #     value_losses.append((torch.tensor([R])-value).pow(2).mean())
    #
    # # reset gradients
    # optimizer.zero_grad()
    #
    # # sum up all the values of policy_losses and value_losses
    # # stacked_policy_loss = policys_losses.sum()
    # # stacked_value_loss = values_losses.sum()
    #
    # stacked_policy_loss = torch.stack(policy_losses).sum()
    # stacked_value_loss = torch.stack(value_losses).sum()
    #
    #
    # loss = stacked_policy_loss + stacked_value_loss
    #
    # print(f"Samples: {len(rewards)} Loss: {loss.item()} - policy: {stacked_policy_loss.item()} - value: {stacked_value_loss.item()} - avg_return: {returns.mean()}")
    #
    # # perform backprop
    # loss.backward()
    # optimizer.step()

def plot_model_history(train_rewards, test_rewards, iter):
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.plot(train_rewards)
    plt.plot(test_rewards)

    plt.title('Avg rewards per episode')
    plt.ylabel('Avg Reward')
    plt.xlabel('Iteration')
    plt.legend(['Train', 'Test'], loc='upper left')
    if not os.path.exists("./ppo/plots"):
        os.mkdir("./ppo/plots")

    plt.savefig(f"./ppo/plots/acc_{iter}.png")
    plt.close()

test_rewards = []
train_rewards = []


def worker_play(args):
    dp, det, seed, repl = args
    play(
        data_path=dp,
        deterministic=det,
        seed=seed,
        replay=repl
    )

import multiprocessing
from multiprocessing import Pool

pool = Pool(multiprocessing.cpu_count()//2)


while iters < max_iters:
    t0 = time.time()
    DATA_PATH = f'ppo/data/{iters}'
    print(DATA_PATH)

    # for x in tqdm.tqdm(range(ppo_steps)):
    for _ in tqdm.tqdm(pool.imap_unordered(worker_play, [(DATA_PATH, 0, None, False) for x in range(ppo_steps)]), total=ppo_steps):
        pass

    # play(data_path=DATA_PATH, deterministic=1, seed=1)

    states, actions, rewards, dones = load_samples(DATA_PATH)

    avg_reward = np.mean(rewards)
    train_rewards.append(np.round(avg_reward, 2))
    if avg_reward > best_train_reward:
        best_train_reward = avg_reward

    print(f'total train reward={avg_reward}, current best={best_train_reward}, rewards={train_rewards[-5:]}')


    numpy_states =  np.array(states)
    states = torch.from_numpy(np.array(states)).double().transpose(1, 3)
    actions = torch.from_numpy(np.array(actions))
    rewards = np.array(rewards)
    dones = np.array(dones)

    finish_episode(states, actions, rewards, dones)


    current_actor = f"./ppo/models/actor_critic_{iters}_pytorch.model"
    torch.save(model, current_actor)
    #
    TEST_PATH = DATA_PATH+"/test"
    if not os.path.exists(TEST_PATH):
        os.mkdir(TEST_PATH)

    for _ in tqdm.tqdm(pool.imap_unordered(worker_play, [(TEST_PATH, 1, x, True) for x in range(ppo_steps)]), total=test_steps):
        pass

    states, actions, rewards, dones = load_samples(TEST_PATH)
    avg_reward = np.mean(rewards)
    #
    test_rewards.append(np.round(avg_reward, 2))
    if avg_reward > best_test_reward:
        current_actor = './ppo/models/model_actor_{}_{}.hdf5'.format(iters, avg_reward)
        torch.save(model, current_actor)
        best_test_reward = avg_reward

    print(f'total test reward={avg_reward}, current best={best_test_reward}, rewards={test_rewards[-5:]} - time={t0-time.time()}')
    iters += 1
    if iters % 10 == 0:
        plot_model_history(train_rewards, test_rewards, iters)

    shutil.rmtree(DATA_PATH)

