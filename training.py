import os
import glob
import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, GlobalAveragePooling2D, LSTM
from tensorflow.keras.models import Model

import numpy as np


def plot_model_history(history):
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f"{EXPERIMENT_PATH}/EPOCHS/{epoch}/training_acc.png")
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f"{EXPERIMENT_PATH}/EPOCHS/{epoch}/training_loss.png")
    plt.close()


def create_model(input_size, output_size):

    learning_rate = 0.001

    x_input = Input(shape=(input_size))
    x1 = Conv2D(32, (1, 1), strides=1, activation='linear', padding="same", data_format='channels_last')(x_input)
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

    x1 = Flatten()(x1)

    x = Dense(128, activation='relu')(x1)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(output_size, activation='softmax')(x)
    # output = Dense(output_size, activation='linear')(x)
    model = Model(x_input, output)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate)
    )
    # model.compile(
    #     loss="mae",
    #     metrics=['accuracy'],
    #     optimizer=tf.keras.optimizers.Adam(lr=learning_rate)
    # )


    return model


NUM_FEATURES = 6
FIELD_SIZE = 8
SEED = 123
STARTING_EPSILON = 1.0

INPUT_SHAPE = (FIELD_SIZE, FIELD_SIZE, NUM_FEATURES)
OUTPUT_SIZE = 5

EPSILON_DECREASE = 0.9
EPSILON = STARTING_EPSILON
NUM_GAMES = 0
NUM_EPOCHS = 2
FUTURE_DISCOUNT = 0.95

EXPERIMENT_NAME = "smaller_network-rl"
model = create_model(INPUT_SHAPE, OUTPUT_SIZE)

EXPERIMENT_PATH = f"./experiments/{EXPERIMENT_NAME}"

if not os.path.exists(EXPERIMENT_PATH):
    os.mkdir(EXPERIMENT_PATH)
MODEL_PATH = f"{EXPERIMENT_PATH}/model"

DATA_PATH = f"./data/single_ship/"
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

SELFPLAY_DATA_PATH = DATA_PATH + "selfplay/0.1"
TRAINING_DATA_PATH = DATA_PATH + "training"
BEATRULE_DATA_PATH = DATA_PATH + "beat_rule/0.0"
BEATRULE_VALIDATION_DATA_PATH = DATA_PATH + "beat_rule_validate/0.0"
VALIDATION_DATA_PATH = DATA_PATH + "validation"
SEED_DATA_PATH = DATA_PATH + "seed"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = create_model(INPUT_SHAPE, OUTPUT_SIZE)
    #from: https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    with open(f"{EXPERIMENT_PATH}/model_summary.txt", 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    print(model.summary())
    model.save(MODEL_PATH)
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=f'{EXPERIMENT_PATH}/model.png')

# out = Dense(5, activation='linear', name="rl-out")(model.layers[-2].output)
# model = Model(model.layers[0].input, out)
# model.compile(
#     loss="mae",
#     metrics=['accuracy'],
#     optimizer=tf.keras.optimizers.RMSProp(clipvalue=1)
# )

# from multiprocessing import Pool
#
# pool = Pool(4)
#
# def run(x):
#     import os
#       os.system(x)


def load_samples(path, max_samples=None):
    total_reward = 0
    states, actions, rewards, dones = [], [], [], []

    files = [x for x in glob.glob(path+"/*_states.npz")]
    if max_samples:
        print(path, len(files), "clipping to", max_samples)
        import random
        files = random.sample(files, max_samples)
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

    print(f"Epoch: {epoch}, num_files={len(states)}, total_reward={total_reward}, avg_reward={round(total_reward/len(states), 2)}")

    train_x, train_y = [], []
    losses = []

    cnt =0
    for episode_states, episode_actions, episode_rewards, episode_dones in tqdm.tqdm(zip(states, actions, rewards, dones)):
        current_qs = model.predict(episode_states)
        for state, current_q, future_q, action_index, reward, done in zip(episode_states, current_qs, current_qs[1:], episode_actions, episode_rewards, episode_dones):
            cnt+=1

            new_q_value = reward
            if not done:
                new_q_value = reward + FUTURE_DISCOUNT * future_q[np.argmax(future_q)]


            new_q = np.array([x for x in current_q])
            new_q[action_index] = new_q_value
            train_x.append(state)
            train_y.append(new_q)

        # for state, action_index in zip(episode_states, episode_actions):
        #     train_x.append(state)
        #     train_y.append(action_index)
        #
        #     #POSSIBLE_MOVES = [Direction.North, Direction.East, Direction.South, Direction.West, Direction.Still]
        #     train_x.append(np.flip(state, 1))
        #     if action_index == 1:
        #         train_y.append(3)
        #     elif action_index == 3:
        #         train_y.append(1)
        #     else:
        #         train_y.append(action_index)
        #
        #     train_x.append(np.flip(state, 0))
        #     if action_index == 0:
        #         train_y.append(2)
        #     elif action_index == 2:
        #         train_y.append(0)
        #     else:
        #         train_y.append(action_index)
        #
        #     train_x.append(np.flip(np.flip(state, 0), 1))
        #     if action_index == 0:
        #         train_y.append(2)
        #     elif action_index == 2:
        #         train_y.append(0)
        #     elif action_index == 1:
        #         train_y.append(3)
        #     elif action_index == 3:
        #         train_y.append(1)
        #     else:
        #         train_y.append(action_index)

    return train_x, train_y


def self_play_games(num_games):
    # f'--seed {SEED} ' \
    command = './halite ' \
              '--replay-directory replays/ ' \
              '--no-timeout ' \
              '--no-logs -v ' \
              '--no-replay ' \
              f'--width {FIELD_SIZE} ' \
              f'--height {FIELD_SIZE} ' \
              f'"python3 NNBot.py {EPSILON} {MODEL_PATH} {DATA_PATH}" ' \
              f'"python3 NNBot.py {EPSILON} {MODEL_PATH} {DATA_PATH}" '

    print(command)
    for x in tqdm.tqdm(range(num_games)):
        os.system(command)


start_epoch = len(list(glob.glob(f"{EXPERIMENT_PATH}/EPOCHS/*")))

if NUM_EPOCHS<=start_epoch:
    print(f"NUM-EPOCHS too small, adjusting, (was={NUM_EPOCHS}, now={start_epoch+1})")
    NUM_EPOCHS=start_epoch+1

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"Starting Epoch {epoch}")
    #self_play_games(NUM_GAMES)

    for i in range(10):

        train_x, train_y = load_samples(TRAINING_DATA_PATH, max_samples=300)
        # test_x, test_y = load_samples(VALIDATION_DATA_PATH)
        # train_x, train_y = load_samples(BEATRULE_DATA_PATH)
        # test_x, test_y = load_samples(BEATRULE_VALIDATION_DATA_PATH)
        # train_x_2, train_y_2 = load_samples(SELFPLAY_DATA_PATH)
        # train_x = np.concatenate([train_x, train_x_2])
        # train_y = np.concatenate([train_y, train_y_2])


        # unique, counts = np.unique(train_y, return_counts=True)
        # weights = dict(zip(unique, np.max(counts)/counts))
        # print(weights)

        # earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


        # def scheduler(epoch):
        #     if epoch < 6:
        #         return 0.0001
        #     else:
        #         return 0.0001 * tf.math.exp(0.1 * (6 - epoch))


        # lrsheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        history = model.fit(
            np.array(train_x),
            np.array(train_y),
            epochs=20,
            shuffle=True,
            verbose=1,
            batch_size=512,
            # class_weight=weights,
            # callbacks=[earlystopper],
            # validation_data=(np.array(test_x), np.array(test_y))
        )

    model.save(MODEL_PATH)
    model.save(f"{EXPERIMENT_PATH}/EPOCHS/{epoch}/model")

    plot_model_history(history)
    EPSILON *= EPSILON_DECREASE



