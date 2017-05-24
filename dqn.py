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
# import theano

VERBOS=False
INTERVENTION_WATCH = True
MODEL_NAME = "Anna" # represents the subdirectory name
MODEL_VERSION = 2 # the version is used in the model name
VERSION_UPDATE = 1200 # new version interval, in seconds
SAVEINVERVAL = 1000 # save interval in iterations
REPORT_INTERVAL = 1000 # frames before reporting, if not verbose
THROTTLING_PERIOD = 2 # frames to skip before training again
ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate expected reward confidence
OBSERVE_PERIOD = 3200. # frames to observe before training
EXPLORE_PERIOD = 3000000. # iterations over which to anneal epsilon from initial to final
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of training_batch to train on
FRAMES_PER_ACTION = 1
LEARNING_RATE = 1e-4
FRAMES_PER_SAMPLE = 4 # How many frames to stack per sample, good for detecting time amounts like velocity

img_rows , img_cols = 80, 80

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
    # 0 nothing
    # 1 right
    # 2 left
    # 3 down
    # 4 up
    # 5 enter
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("Done building.")
    return model

def exerciseNetwork(model):

    # store the previous OBSERVE_PERIODs in replay memory
    D = deque()

    # the first state is doing nothing
    # and preprocess the image to 80x80x4
    x_current, r_0, terminal, tact = pacman.step(0)

    x_current = skimage.color.rgb2gray(x_current)
    x_current = skimage.transform.resize(x_current,(80,80))
    x_current = skimage.exposure.rescale_intensity(x_current,out_range=(0,255))

    s_current = np.stack((x_current, x_current, x_current, x_current), axis=2)
    # print (s_current.shape)

    # reshape to match Keras requirements
    s_current = s_current.reshape(1, s_current.shape[0], s_current.shape[1], s_current.shape[2])  #1*80*80*4


    path = ".\\"+MODEL_NAME+"\\"+str(MODEL_VERSION)
    if mode == 1:
        MUSTLOAD = True
        epsilon = 0.01;
        print("Exercise mode: Running")
    else:
        MUSTLOAD = False
        epsilon = INITIAL_EPSILON
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
            print("Starting a fresh model.")

    t = 0
    last_version_change = datetime.now();
    while (True):
        action_index = 0
        reward = 0
        # choose an action epsilon greedy
        # this is for exploration over exploitation
        if t % FRAMES_PER_ACTION == 0:
            if random.random() <= epsilon:
                if(VERBOS):
                    print("*********************Random Action*********************")
                action_index = random.randrange(ACTIONS)
            else:
                q = model.predict(s_current)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q

        # We reduced the epsilon gradually
        # The closer we get to a pro player, the less we should EXPLORE_PERIOD random strategies
        # We will reach the final value after EXPLORE_PERIOD iterations
        if epsilon > FINAL_EPSILON and t > OBSERVE_PERIOD:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_PERIOD

        #run the selected action and observed next state and reward from game
        #next state reward gameover?
        x_next_colored, reward, terminal, tact = pacman.step(action_index)

        if INTERVENTION_WATCH & (tact != action_index):
            action_index = tact
            print("User action: ", action_index)
        
        if(terminal):
            print("Gameover. ",MODEL_NAME, " scored: ", pacman.GetScore())

        # preprocess first to remove extra data and highlight contours
        x_next = skimage.color.rgb2gray(x_next_colored)
        x_next = skimage.transform.resize(x_next,(80,80))
        x_next = skimage.exposure.rescale_intensity(x_next, out_range=(0, 255))

        x_next = x_next.reshape(1, x_next.shape[0], x_next.shape[1], 1) #1x80x80x1 the first dimension is 1 for keras
        s_next = np.append(x_next, s_current[:, :, :, :3], axis=3) # take 3 frames from previous s and add the new one

        if(mode == 0): # training
            total_loss = 0
            Q_sa_next = 0
            # store the transition in D
            # s,a,r,s',t
            D.append((s_current, action_index, reward, s_next, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # only train if done observing, this is because our network needs a lot of data to not over fit
            if (t > OBSERVE_PERIOD) & (t % THROTTLING_PERIOD == 0):
                # pick a random sample from the replay memory to train on
                training_batch = random.sample(D, BATCH)

                inputs = np.zeros((BATCH, s_current.shape[1], s_current.shape[2], s_current.shape[3]))   #32, 80, 80, 4

                targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

                # Now we do the experience replay
                for i in range(0, len(training_batch)):
                    # training_batch[i] contains  s,a,r,s',t
                    state_current = training_batch[i][0]
                    action_t = training_batch[i][1]   #This is action index
                    reward_t = training_batch[i][2]
                    state_next = training_batch[i][3]
                    terminal = training_batch[i][4]
                    # if terminated, only equals reward

                    inputs[i:i + 1] = state_current

                    targets[i] = model.predict(state_current)  # Hitting each button probability
                    Q_sa_next = model.predict(state_next) # we use the current model to get Q(s',a')

                    if terminal:
                        targets[i, action_t] = reward_t # the reward is not going to increase, game is over
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa_next) # predict the expected reward increase

                total_loss += model.train_on_batch(inputs, targets)

                # version change
                if ((datetime.now() - last_version_change).seconds > VERSION_UPDATE):
                    global MODEL_VERSION
                    MODEL_VERSION = MODEL_VERSION + 1;
                    last_version_change = datetime.now();

                # save progress
                if t % (SAVEINVERVAL) == 0:
                    print("Saving model...")
                    directory = ".\\" + MODEL_NAME
                    if (not os.path.exists(directory)):
                        os.makedirs(directory)
                    path = directory + "\\" + str(MODEL_VERSION);
                    model.save_weights(path, overwrite=True)
                    # dump the model structure info for future reference
                    with open(directory + "\\model.json", "w") as outfile:
                        json.dump(model.to_json(), outfile)
                    print("Saved to: " + path);
            if VERBOS:
                print("Action: ", action_index, "Reward:", reward, "Terminal:", terminal)

        s_current = s_next

        t = t + 1

        if(VERBOS | (t%REPORT_INTERVAL == 0)):
            # print info
            state = ""
            if mode == 1:
                staet = "Running"
            elif t <= OBSERVE_PERIOD:
                state = "Observing"
            elif t > OBSERVE_PERIOD and t <= OBSERVE_PERIOD + EXPLORE_PERIOD:
                state = "Exploring"
            else:
                state = "Training"
            if mode == 1:
                if (mode == 1):
                 print("VERSION", MODEL_VERSION, "STAMP", t, "/ SCORE", pacman.GetScore(), "/ STATE", state)
            elif mode == 0:
                print("VERSION", MODEL_VERSION, "STAMP", t, "/ SCORE", pacman.GetScore(), "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
                "/ Q_MAX " , np.max(Q_sa_next), "/ Loss ", total_loss)

def startWork():
    model = buildmodel()
    exerciseNetwork(model)

def main():
    parser = argparse.ArgumentParser(description='Train an AI to play Pacman')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('-v',action='store_true')
    parser.add_argument('-iw', action='store_true') # allowing and accounting for human intervention, enriches the meaningful data
    args = vars(parser.parse_args())
    global VERBOS, INTERVENTION_WATCH, mode
    VERBOS = args['v']
    INTERVENTION_WATCH = args['iw']
    if args['mode'] == 'Run':
        mode = 1
    elif args['mode'] == 'Train':
        mode = 0
    startWork()

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
