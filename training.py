import os
import tensorflow as tf

def create_model(input_size, output_size):
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, GlobalAveragePooling2D, LSTM
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    import tensorflow.keras.backend as K

    learning_rate = 0.01

    x_input = layers.Input(shape=(input_size))
    x1 = Conv2D(64, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(x_input)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x2 = Conv2D(64, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x3 = Conv2D(64, (3, 3), strides=1, activation='linear', padding="same", data_format='channels_last')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    x3 = Flatten()(x3)
    x = Dense(128, activation='relu')(x3)
    x = Dense(128, activation='relu')(x)
    output = Dense(output_size, activation='linear')(x)
    model = Model(x_input, output)
    model.compile(
        loss='mse',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate)
    )
    print(input_size, output_size)
    print(model.summary())


    return model

EXPERIMENT_NAME = "first_try"
EXPERIMENT_PATH = f"./experiments/{EXPERIMENT_NAME}"

if not os.path.exists(EXPERIMENT_PATH):
    os.mkdir(EXPERIMENT_PATH)
MODEL_PATH = f"{EXPERIMENT_PATH}/model"

TRAINING_DATA_PATH = f"{EXPERIMENT_PATH}/training/"
if not os.path.exists(TRAINING_DATA_PATH):
    os.mkdir(TRAINING_DATA_PATH)


NUM_FEATURES = 5
FIELD_SIZE = 8
SEED = 123
STARTING_EPSILON = 0.0

INPUT_SHAPE = (NUM_FEATURES, FIELD_SIZE, FIELD_SIZE)
OUTPUT_SIZE = 5

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = create_model(INPUT_SHAPE, OUTPUT_SIZE)
    #from: https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    with open(f"{EXPERIMENT_PATH}/model_summary.txt", 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    model.save(MODEL_PATH)


command = './halite ' \
          f'--seed {SEED} ' \
          '--replay-directory replays/ ' \
          '--turn-limit 10 ' \
          '--no-timeout ' \
          '--no-logs -vvv ' \
          f'--width {FIELD_SIZE} ' \
          f'--height {FIELD_SIZE} ' \
          f'"python3 NNBot.py {STARTING_EPSILON} {MODEL_PATH} {TRAINING_DATA_PATH}" ' \
          f'"python3 NNBot.py {STARTING_EPSILON} {MODEL_PATH} {TRAINING_DATA_PATH}" '

print(command)
os.system(command)

