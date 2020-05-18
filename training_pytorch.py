import os
import glob
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch import optim

import numpy as np

def createBatchNorm(layer):
    return nn.BatchNorm2d(layer.out_channels)


class MyModel(nn.Module):
    def __init__(self, size_x, size_y):
        super().__init__()

        self.cnn1 = nn.Conv2d(
            in_channels=6,
            out_channels=32,
            kernel_size=(1, 1),
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.cnns = []
        for i in range(5):
            self.cnns.extend(
                [
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=(3, 3),
                        stride=1,
                        padding=1,
                        bias=False
                    ),
                    F.relu,
                    nn.BatchNorm2d(32)
                ]
            )

        self.dense1 = nn.Linear(
            in_features=32*8*8,
            out_features=128,
            bias=False
        )

        self.dense2 = nn.Linear(
            in_features=128,
            out_features=5,
            bias=True
        )

    def forward(self, input_image):
        x = self.bn1(F.relu(self.cnn1(input_image)))

        for cnn in self.cnns:
            x = cnn(x)

        x = x.view(-1, 32*8*8)

        x = F.dropout(F.relu(self.dense1(x)), 0.5)

        return self.dense2(x)

# def create_model(input_size, output_size):
#     from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add
#     from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, GlobalAveragePooling2D, LSTM
#     from tensorflow.keras import layers
#     from tensorflow.keras.models import Model
#
#     learning_rate = 0.001
#
#     x_input = layers.Input(shape=(input_size))
#     x1 = Conv2D(32, (1, 1), strides=1, activation='linear', padding="same", data_format='channels_last')(x_input)
#     x1 = Activation("relu")(x1)
#     layer1_out = BatchNormalization()(x1)
#
#     x1 = Conv2D(32, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(layer1_out)
#     x1 = Activation("relu")(x1)
#     x1 = BatchNormalization()(x1)
#
#     prev = x1
#     for i in range(5):
#         x1 = Conv2D(32, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(prev)
#         x1 = Activation("relu")(x1)
#         x1 = BatchNormalization()(x1)
#         prev = x1
#
#     x1 = Flatten()(x1)
#
#     x = Dense(128, activation='relu')(x1)
#     x = Dropout(0.4)(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.4)(x)
#     output = Dense(output_size, activation='softmax')(x)
#     model = Model(x_input, output)
#     model.compile(
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#         metrics=['accuracy'],
#         optimizer=tf.keras.optimizers.Adam(lr=learning_rate)
#     )
#
#     return model



DATA_PATH = f"./data/single_ship/"
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

SELFPLAY_DATA_PATH = DATA_PATH + "selfplay/0.1"
TRAINING_DATA_PATH = DATA_PATH + "training"
VALIDATION_DATA_PATH = DATA_PATH + "validation"
SEED_DATA_PATH = DATA_PATH + "seed"


def load_samples(path, max_samples=None):
    total_reward = 0
    states, actions, rewards, dones = [], [], [], []

    files = [x for x in glob.glob(path+"/*_states.npz")]
    if max_samples:
        print(path, len(files), "clipping to", max_samples)
        files = files[:max_samples]
    else:
        print(path, len(files))

    for file in tqdm.tqdm(files):
        total_reward += int(file.split('/')[-1].split('_')[0])

        try:
            BASE_PATH = file.replace("_states.npz", '')
            states.append(np.load(f"{BASE_PATH}_states.npz", allow_pickle=True)["arr_0"])
            actions.append(np.load(f"{BASE_PATH}_actions.npz", allow_pickle=True)["arr_0"])
            rewards.append(np.load(f"{BASE_PATH}_rewards.npz", allow_pickle=True)["arr_0"])
            dones.append(np.load(f"{BASE_PATH}_dones.npz", allow_pickle=True)["arr_0"])
        except Exception as e:
            print(file, "error:", e)


    train_x, train_y = [], []
    losses = []


    for episode_states, episode_actions, episode_rewards, episode_dones in tqdm.tqdm(zip(states, actions, rewards, dones)):
        # current_qs = model.predict(episode_states)
        # for state, current_q, future_q, action_index, reward, done in zip(episode_states, current_qs, current_qs[1:], episode_actions, episode_rewards, episode_dones):
        #
        #     new_q_value = reward
        #     if not done:
        #         new_q_value = reward + FUTURE_DISCOUNT * future_q[np.argmax(future_q)]
        #
        #
        #     new_q = np.array([x for x in current_q])
        #     new_q[action_index] = new_q_value

        for state, action_index in zip(episode_states, episode_actions):
            train_x.append(state) #(100, 32, 32, 6)
            train_y.append(action_index) #[1,2,3,5...] (100, 1)

    return np.array(train_x), np.array(train_y)


train_x, train_y = load_samples(TRAINING_DATA_PATH, max_samples=1)

train_x_tensor = torch.Tensor(train_x).transpose(1, 3)
train_y_tensor = torch.LongTensor(train_y)

myData = TensorDataset(train_x_tensor, train_y_tensor)
myDataLoader = DataLoader(myData, batch_size=512, shuffle=True)


#test_x, test_y = load_samples(VALIDATION_DATA_PATH)
# unique, counts = np.unique(train_y, return_counts=True)
# weights = dict(zip(unique, np.max(counts)/counts))
# print(weights)

model = MyModel(8, 8)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

print(train_x.shape)
for epoch in range(0, 100):

    losses = []
    for X, y in myDataLoader:
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if epoch%25==0:
        print(f"Ending Epoch {epoch}, {np.mean(losses)}")


print(F.softmax(model(X), dim=1))
torch.save(model, "pytorch.model")
