#!/usr/bin/env python
from __future__ import print_function

import game.pacman_container as pacman
import argparse
import skimage as skimage
from skimage import transform, color, exposure
import random
import numpy as np
from collections import deque
import os.path
from datetime import datetime

import json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
# Import theano

MODEL_NAME = "Anna" # An AI has to have a name! Also the subdirectory name
MODEL_VERSION = 54 # The version is used for the file name
VERSION_UPDATE = 1200 # New version interval, in seconds
SAVE_INTERVAL = 1000 # Save interval in iterations
REPORT_INTERVAL = 1000 # Frames before reporting, if not verbose
THROTTLING_PERIOD = 2 # Frames to skip before training again
ACTIONS = 6 # Number of valid actions
INITIAL_GAMMA = 0.5 # Low confidence in predictions while exploring
# A large gamma for a game where positive rewards are common to be subsequent, like pacman here, can eventually cause an overflow
FINAL_GAMMA = 0.7 # High confidence when perfecting the technique
OBSERVE_PERIOD = 6400 # Frames to observe before training
EXPLORE_PERIOD = 3000000. # Iterations over which to anneal EPSILON from initial to final
FINAL_EPSILON = 0.001 # Final value of EPSILON
INITIAL_EPSILON = 0.1 # Starting value of EPSILON
REPLAY_MEMORY = 50000 # Number of previous transitions to remember
BATCH = 32 # Size of experiences to train on
FRAMES_PER_ACTION = 1
LEARNING_RATE = 1e-5
FRAMES_PER_SAMPLE = 4 # How many frames to stack per sample, good for detecting time amounts like velocity

img_rows , img_cols = 80, 80

amap = ["Nothing", "Right", "Left", "Down", "Up", "Enter"]

def buildmodel():
    print("Building the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,FRAMES_PER_SAMPLE)))  #80*80*4
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

def doAction(action, s_current):
    # Run the selected action and observed next state and reward from game
    # Next state reward gameover?
    x_next_colored, reward, terminal, tact = pacman.step(action)

    if INTERVENTION_WATCH & (tact != action):
        action = tact
        print("User action: ", action)

    if (pacman.IsGameOver()):
        print("Gameover. ", MODEL_NAME, " scored: ", pacman.GetScore())

    # Preprocess first to remove extra data and highlight contours
    x_next = skimage.color.rgb2gray(x_next_colored)
    x_next = skimage.transform.resize(x_next, (80, 80))
    x_next = skimage.exposure.rescale_intensity(x_next, out_range=(0, 255))

    x_next = x_next.reshape(1, x_next.shape[0], x_next.shape[1], 1)  # 1x80x80x1 the first dimension is 1 for keras

    s_next = np.append(x_next, s_current[:, :, :, :3], axis=3)  # Take 3 frames from previous s and add the new one

    return s_next, reward, terminal, action


def exerciseNetwork(model):

    # Store the previous observations in replay memory
    RECORD = deque()

    # The first two states are doing nothing
    # Preprocess the image to 80x80x4 and repeat 4 tims for the first state
    x_prev, r_0, t_0, tact = pacman.step(0)

    x_prev = skimage.color.rgb2gray(x_prev)
    x_prev = skimage.transform.resize(x_prev,(80,80))
    x_prev = skimage.exposure.rescale_intensity(x_prev,out_range=(0,255))

    s_prev = np.stack((x_prev, x_prev, x_prev, x_prev), axis=2)
    # Print (s_prev.shape)

    # Reshape to match Keras requirements
    s_prev = s_prev.reshape(1, s_prev.shape[0], s_prev.shape[1], s_prev.shape[2])  #1*80*80*4

    s_current, r_prev, t_current, a_prev = doAction(0, s_prev)


    path = ".\\"+MODEL_NAME+"\\"+str(MODEL_VERSION)
    if PROG_MODE == 1:
        MUSTLOAD = True
        EPSILON = FINAL_EPSILON;
        print("Exercise mode: Running")
    else:
        MUSTLOAD = False
        EPSILON = INITIAL_EPSILON
        GAMMA = INITIAL_GAMMA
        print("Exercise mode: Training")

    if (os.path.exists(path)):
        print("Loading weights for model " + MODEL_NAME + " version " + str(MODEL_VERSION))
        model.load_weights(path)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("Weights loaded")
    else:
        if(MUSTLOAD):
            print("File not found: " + path);
            exit(-1);
        else:
            print("Starting a fresh model...")

    t = 0
    last_version_change = datetime.now();
    while (True):
        action_index = 0
        max_q = 0
        # Choose an action EPSILON greedy
        # This is for exploration over exploitation
        if t % FRAMES_PER_ACTION == 0:
            q = model.predict(s_current)  # Input a stack of 4 images, get the prediction
            max_q = np.argmax(q)
            if random.random() <= EPSILON:
                action_index = random.randrange(ACTIONS)
            else:
                action_index = max_q

        # We reduced the EPSILON gradually
        # The closer we get to a pro player, the less we should EXPLORE_PERIOD random strategies
        # We will reach the final value after EXPLORE_PERIOD iterations
        if t > OBSERVE_PERIOD:
            if EPSILON > FINAL_EPSILON:
                EPSILON += (FINAL_EPSILON - INITIAL_EPSILON) / EXPLORE_PERIOD
            if GAMMA < FINAL_GAMMA:
                GAMMA += (FINAL_GAMMA - INITIAL_GAMMA) / EXPLORE_PERIOD

        s_next, reward, terminal, action_index = doAction(action_index, s_current)

        if(PROG_MODE == 0): # Training
            total_loss = 0

            # Version change
            if ((datetime.now() - last_version_change).seconds > VERSION_UPDATE):
                global MODEL_VERSION
                MODEL_VERSION += 1
                last_version_change = datetime.now()

            # Store the experience in D
            RECORD.append((s_prev, a_prev, r_prev, t_current, s_current, action_index, reward, terminal, s_next))
            if len(RECORD) > REPLAY_MEMORY:
                RECORD.popleft()

            # Only train if done observing, this is because our network needs a lot of data to not over fit
            if (t > OBSERVE_PERIOD) & (t % THROTTLING_PERIOD == 0):
                # Pick a random sample from the replay memory to train on
                experiences = random.sample(RECORD, BATCH)

                inputs = np.zeros((BATCH*2, s_current.shape[1], s_current.shape[2], s_current.shape[3]))   #32, 80, 80, 4

                targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

                # Now we do the experience replay
                for i in range(0, BATCH):
                    # An experience contains states in the order t,s,a,r minus t for the first state and only t,s for the last
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

                total_loss += model.train_on_batch(inputs, targets)

                # Save progress
                if t % (SAVE_INTERVAL) == 0:
                    print("Saving model...")
                    directory = ".\\" + MODEL_NAME
                    if (not os.path.exists(directory)):
                        os.makedirs(directory)
                    path = directory + "\\" + str(MODEL_VERSION)
                    model.save_weights(path, overwrite=True)
                    # Dump the model structure info for future reference
                    with open(directory + "\\model.json", "w") as outfile:
                        json.dump(model.to_json(), outfile)
                    print("Saved to: " + path);

        #RECORD.append((s_prev, a_prev, r_prev, t_current, action_index, reward, s_next, terminal))
        s_prev = s_current
        s_current = s_next
        a_prev = action_index
        r_prev = reward
        t_current = terminal

        t = t + 1

        if VERBOSE | (t%REPORT_INTERVAL == 0) | (SIGRW & (np.abs(reward) >= 0.1)):
            # Print info
            if PROG_MODE == 1:
                state = "Running"
            elif t <= OBSERVE_PERIOD:
                state = "Observing"
            elif t > OBSERVE_PERIOD and t <= OBSERVE_PERIOD + EXPLORE_PERIOD:
                state = "Exploring"
            else:
                state = "Training"
            if PROG_MODE == 1:
                print("VERSION", MODEL_VERSION, "STAMP", t, "/ SCORE", pacman.GetScore(), "/ STATE", state, \
                      "/ ACTION", amap[action_index], "/ REWARD", reward, "/ EXPECTED", \
                      q[0][action_index], "/ BEST ", amap[max_q], "/ EXPECTED", q[0][max_q], \
                      "/ EPSILON", EPSILON)
            elif PROG_MODE == 0:
                print("VERSION", MODEL_VERSION, "STAMP", t, "/ SCORE", pacman.GetScore(), "/ STATE", state, \
                      "/ ACTION", amap[action_index], "/ REWARD", reward, "/ EXPECTED", \
                      q[0][action_index], "/ BEST " , amap[max_q],"/ EXPECTED", q[0][max_q], "/ Loss ", total_loss, \
                      "/ EPSILON", EPSILON, "/ GAMMA", GAMMA)

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
