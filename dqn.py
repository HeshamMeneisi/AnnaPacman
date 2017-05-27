from __future__ import print_function
import game.pacman_container as pacman
import argparse, random, os.path, json
import skimage
from skimage import transform, color, exposure
import numpy as np
from collections import deque
from datetime import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
# import pylab as plt
# import theano

MODEL_NAME = "Anna"         # An AI has to have a name! Also the subdirectory name
MODEL_VERSION = 100         # The version is used for the file name
VERSION_UPDATE = 1200       # New version interval, in seconds
SAVE_INTERVAL = 1000        # Save interval in iterations
REPORT_INTERVAL = 1000      # Frames before reporting, if not verbose
THROTTLING_PERIOD = 2       # Frames to skip before training again
ACTIONS = 6                 # Number of valid actions
INITIAL_GAMMA = 0.6         # Low confidence in predictions while exploring
# ** A large gamma for a game where positive rewards are usually subsequent can eventually cause an overflow
FINAL_GAMMA = 0.8           # High confidence when perfecting the technique
OBSERVE_PERIOD = 6400       # Frames to observe before training
EXPLORE_PERIOD = 1000000    # Iterations over which to anneal EPSILON from initial to final
FINAL_EPSILON = 0.001       # Final value of EPSILON
INITIAL_EPSILON = 0.1       # Starting value of EPSILON
REPLAY_MEMORY = 50000       # Number of previous transitions to remember
BATCH = 32                  # Size of experiences to train on
FRAMES_PER_ACTION = 1       # The delay before taking another action
LEARNING_RATE = 1e-4        # Our network's learning rate
FRAMES_PER_SAMPLE = 4       # How many frames to stack per sample, good for detecting time-based amounts like velocity

# ** The image size, images are rotated 90 degrees in a matrix so the height is rows and the width is columns
# ** It's better to use a square image because otherwise lines might get jagged or completely disappear while resizing
IMG_ROWS , IMG_COLS = 132, 132

# A map of all action names for logging
AMAP = ['None', 'Right', 'Left', 'Down', 'Up', 'Enter']

# Paths
MODEL_DIR = ".\\" + MODEL_NAME
SAVE_PATH = lambda: MODEL_DIR + "\\" + str(MODEL_VERSION)
STATE_PATH = MODEL_DIR + "\\state"

# When training, the state is reset to observing on restart. No memory dumping yet.
STATE_LABELS = ['Testing', 'Observing', 'Exploring', 'Training']

# def displayImage(img):
#     # This code would display the image identical to the game itself
#     # However, orientation and/or mirroring will not affect the training
#     # This is mainly because there's no sense of directions in the network when initialized
#     # For demonstration, this is similar to how babies see the world upside down for the first few days
#     plt.imshow(np.fliplr(skimage.transform.rotate(skimage.exposure.rescale_intensity(
#         skimage.transform.resize(img, (IMG_ROWS, IMG_COLS)), (0, 255),-90))), cmap='gray')
#     print("Waiting...")
#     input()

# Build the model
# I'm so far using the same model that worked for the Keras-FlappyBird project but it might need some modifications
def buildmodel():

    print("Building the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', \
              input_shape=(IMG_ROWS, IMG_COLS, FRAMES_PER_SAMPLE)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(6))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("Done building.")
    return model

# Run the selected action and observe the next state and reward from game
def doAction(action, s_current):

    # Next state reward gameover?
    x_next_colored, reward, terminal, tact = pacman.step(action)

    if INTERVENTION_WATCH & (tact != action):
        action = tact
        print("User action: ", action)

    if (pacman.IsGameOver()):
        print("Gameover. ", MODEL_NAME, " scored: ", pacman.GetScore())

    # Preprocess first to remove extra data and highlight contours
    x_next = skimage.color.rgb2gray(x_next_colored)
    x_next = skimage.transform.resize(x_next, (IMG_ROWS, IMG_COLS))
    x_next = skimage.exposure.rescale_intensity(x_next, out_range=(0, 255))
    x_next = x_next.reshape(1, IMG_ROWS, IMG_COLS, 1)  # First dimension depth is one for keras

    # Take FRAMES_PER_SAMPLE-1 frames from previous s and add the new one
    s_next = np.append(x_next, s_current[:, :, :, :FRAMES_PER_SAMPLE-1], axis=3)

    return s_next, reward, terminal, action

def exerciseNetwork(model):

    t = c = 0  # t is the current iteration, c is a counter for the observation period
    last_version_change = datetime.now()

    # Store the previous observations in replay memory
    RECORD = deque()

    # The first two states are doing nothing
    # Pre-process the image to a IMG_ROWSxIMG_COLS grayscale and repeat FRAMES_PER_SAMPLE times for the first state
    x_prev, r_0, t_0, tact = pacman.step(0)

    x_prev = skimage.color.rgb2gray(x_prev)
    x_prev = skimage.transform.resize(x_prev,(IMG_ROWS,IMG_COLS))
    x_prev = skimage.exposure.rescale_intensity(x_prev,out_range=(0,255))

    # First dimension depth is one for keras
    s_prev = x_prev = x_prev.reshape(1,IMG_ROWS,IMG_COLS,1)

    for i in range(FRAMES_PER_SAMPLE-1):
        s_prev = np.append(s_prev,x_prev,axis=3)

    s_current, r_prev, t_current, a_prev = doAction(0, s_prev)

    if PROG_MODE == 1:
        MUSTLOAD = True
        EPSILON = FINAL_EPSILON;
        print("Exercise mode: Running")
    else:
        MUSTLOAD = False
        EPSILON = INITIAL_EPSILON
        GAMMA = INITIAL_GAMMA
        print("Exercise mode: Training")

    path = SAVE_PATH()
    if (os.path.exists(path)):
        print("Loading weights for model ", MODEL_NAME, " version ", MODEL_VERSION)
        model.load_weights(path)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("Weights loaded")
    else:
        if(MUSTLOAD):
            print("File not found: ", path)
            exit(-1)
        else:
            print("Starting a fresh model...")

    # Look for a saved state
    if os.path.exists(STATE_PATH+".npy"):
        print("Found a state file, loading...")
        [t, MODEL_VERSION, EPSILON, GAMMA] = np.load(STATE_PATH+".npy")
        t = int(t)
        MODEL_VERSION = int(MODEL_VERSION)
        print("State loaded", "T =", t, "Version =", MODEL_VERSION, "EPSILON =", EPSILON, "GAMMA =", GAMMA)

    while (True):
        action_index = 0
        max_q = 0
        loss = 0

        # Update state
        global STATE
        if PROG_MODE == 1:
            STATE = 0
        elif c <= OBSERVE_PERIOD:
            STATE = 1
            c += 1
        elif OBSERVE_PERIOD < t <= OBSERVE_PERIOD + EXPLORE_PERIOD:
            STATE = 2
        else:
            STATE = 3

        # Choose an action
        if t % FRAMES_PER_ACTION == 0:
            q = model.predict(s_current)  # Input a stack of FRAMES_PER_SAMPLE images, get the prediction
            max_q = np.argmax(q)
            # EPSILON is for exploration over exploitation
            if random.random() <= EPSILON:
                action_index = random.randrange(ACTIONS)
            else:
                action_index = max_q

        s_next, reward, terminal, action_index = doAction(action_index, s_current)

        if PROG_MODE == 0: # Training

            # Version change
            if (datetime.now() - last_version_change).seconds > VERSION_UPDATE:
                global MODEL_VERSION
                MODEL_VERSION += 1
                last_version_change = datetime.now()

            # Store the experience in D
            RECORD.append((s_prev, a_prev, r_prev, t_current, s_current, action_index, reward, terminal, s_next))
            if len(RECORD) > REPLAY_MEMORY:
                RECORD.popleft()

            # Only train if done observing, this is because our network needs a lot of data to not over fit or overflow
            if STATE > 1 and t % THROTTLING_PERIOD == 0:

                # Reduced EPSILON gradually
                # The closer we get to a pro player, the less we should explore random strategies
                # We will reach the final value after EXPLORE_PERIOD iterations
                if STATE > 1:
                    if EPSILON > FINAL_EPSILON:
                        EPSILON += (FINAL_EPSILON - INITIAL_EPSILON) / EXPLORE_PERIOD

                # Increase GAMMA gradually
                # The closer we get to a pro player, the more confident we should be with our predictions
                # We will reach the final value after EXPLORE_PERIOD iterations
                if GAMMA < FINAL_GAMMA:
                    GAMMA += (FINAL_GAMMA - INITIAL_GAMMA) / EXPLORE_PERIOD

                # Pick a random sample from the replay memory to train on
                experiences = random.sample(RECORD, BATCH)
                inputs = np.zeros((BATCH*2, IMG_ROWS, IMG_COLS, FRAMES_PER_SAMPLE))
                targets = np.zeros((inputs.shape[0], ACTIONS))

                # Now we do the experience replay
                for i in range(0, BATCH):
                    # An experience contains t,s,a,r for each state
                    # Minus t for the first state, and only t,s for the last
                    state_0 = experiences[i][0]
                    action_0 = experiences[i][1]
                    reward_0 = experiences[i][2]

                    terminal_1 = experiences[i][3]
                    state_1 = experiences[i][4]
                    action_1 = experiences[i][5]
                    reward_1 = experiences[i][6]

                    terminal_2 = experiences[i][7]
                    state_2 = experiences[i][8]

                    inputs[i:i + 1] = state_0
                    inputs[BATCH + i:BATCH + i + 1] = state_1

                    # We use the current model to predict the maximum future reward
                    Q_sa_0 = model.predict(state_0)
                    Q_sa_1 = model.predict(state_1)
                    Q_sa_2 = model.predict(state_2)

                    targets[i] = Q_sa_0
                    targets[BATCH + i] = Q_sa_1

                    reward_1_max = np.max([np.max(Q_sa_1), reward_1]) # Maximum expected or experienced reward
                    reward_2_max = np.max(Q_sa_2)

                    if terminal_1:
                        # The reward is not going to increase, game is over
                        targets[i, action_0] = reward_0

                        # State_1 is not useful, repeat the situation that ended the game
                        inputs[BATCH + i:BATCH + i + 1] = state_0
                        targets[BATCH + i] = Q_sa_0
                        targets[BATCH + i, action_1] = reward_0

                    elif terminal_2:
                        # Predict the expected reward increase
                        targets[i, action_0] = reward_0 + GAMMA * reward_1_max

                        # Game is over for action_1
                        targets[BATCH + i, action_1] = reward_1

                    else:
                        # Predict the expected maximum reward increase after 2 actions
                        targets[i, action_0] = reward_0 + GAMMA * reward_1_max + (GAMMA**5) * reward_2_max

                        targets[BATCH + i, action_1] = reward_1 + GAMMA * reward_2_max

                loss = model.train_on_batch(inputs, targets)

                # Save progress
                if t % SAVE_INTERVAL == 0:
                    print("Saving model...")
                    if not os.path.exists(MODEL_DIR):
                        os.makedirs(MODEL_DIR)

                    # Save the weights
                    model.save_weights(SAVE_PATH(), overwrite=True)

                    # Dump the model structure info for future reference
                    with open(MODEL_DIR + "\\model.json", "w") as outfile:
                        json.dump(model.to_json(), outfile)

                    print("Saved to: ", SAVE_PATH())

                    # Save the state variables
                    np.save(STATE_PATH, [t, MODEL_VERSION, EPSILON, GAMMA])
                    print("State saved to: ", STATE_PATH)

        # Experience: (s_prev, a_prev, r_prev, t_current, action_index, reward, s_next, terminal)
        s_prev = s_current
        s_current = s_next
        a_prev = action_index
        r_prev = reward
        t_current = terminal

        if VERBOSE or t % REPORT_INTERVAL == 0 or (SIGRW and np.abs(reward) >= 0.1):
            # Print info
            if PROG_MODE == 1:
                print("V", MODEL_VERSION, "T", t, "/ S", pacman.GetScore(), "/ ST", STATE_LABELS[STATE], \
                      "/ ACTION (BEST)", AMAP[action_index], "(",AMAP[max_q], ")/ REWARD", reward, "/ EXPECTED", \
                      q[0][action_index],"(", q[0][max_q], ")/ EPSILON", EPSILON)
            elif PROG_MODE == 0:
                print("V", MODEL_VERSION, "T", t, "/ S", pacman.GetScore(), "/ ST", STATE_LABELS[STATE], \
                      "/ ACTION (BEST)", AMAP[action_index], "(",AMAP[max_q], ")/ REWARD", reward, "/ EXPECTED", \
                      q[0][action_index],"(", q[0][max_q], ")", \
                      "/ Loss ", loss,  "/ EPSILON", EPSILON, "/ GAMMA", GAMMA)

        t += 1

def startWork():

    model = buildmodel()
    exerciseNetwork(model)

def main():

    parser = argparse.ArgumentParser(description='Train an AI to play Pacman')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('-v',action='store_true')
    parser.add_argument('-sig', action='store_true') # Display significant rewards
    parser.add_argument('-iw', action='store_true') # Allowing and accounting for human intervention, enriches the meaningful data
    args = vars(parser.parse_args())
    global VERBOSE, INTERVENTION_WATCH, PROG_MODE, SIGRW
    VERBOSE = args['v']
    SIGRW = args['sig']
    INTERVENTION_WATCH = args['iw']
    if args['mode'] == 'Run':
        PROG_MODE = 1
    elif args['mode'] == 'Train':
        PROG_MODE = 0
    startWork()

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
