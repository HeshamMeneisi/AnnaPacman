# Overview
Anna is an AI trained using the DQN (Deep Q-Network) algorithm to play Pacman.

# Dependencies

python 3.5

keras

tensorflow/theano

scikit-image

h5py

GPU Training:
cuda

# Running
To start training, run `dqn.py` with `-m Train`
To test the latest agent, use `-m Test`
The `-iw` flag enables user interactions, these interactions are also recorded as experiences while training.
The `-v` flag enables verbose reporting.

Example:
```sh
python3 dqn.py -m Train -v
```

# Code Structure

### `env_test`

This file checks the environment functinoality. First a check is performed on Keras to verify it exists and retrieve the backend in use. Then, that backend is tested using a basic function and whether the CPU or GPU is being used is reported.

```sh
python3 env_test.py
```

```
## Checking Keras

Using TensorFlow backend.

## Checking TensorFlow

Version 1.5.0

Testing...
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 13656171249690873929
]
2018-02-14 18:32:51.352817: I tensorflow/core/common_runtime/direct_session.cc:297] Device mapping:

Device mapping: no known devices.
Exp: (Exp): /job:localhost/replica:0/task:0/device:CPU:0
x: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2018-02-14 18:32:51.355679: I tensorflow/core/common_runtime/placer.cc:874] Exp: (Exp)/job:localhost/replica:0/task:0/device:CPU:0
2018-02-14 18:32:51.355702: I tensorflow/core/common_runtime/placer.cc:874] x: (Const)/job:localhost/replica:0/task:0/device:CPU:0
Result [1.53210308 1.01887397 2.53835477 ... 1.09205114 1.66782705 2.41976454]
True Values [1.53210308 1.01887397 2.53835477 ... 1.09205114 1.66782705 2.41976454]
Looping 1000 times took 0.012136 seconds
TensorFlow is using the CPU
```

### `dqn`

This is the DQN implementation.

#### Important Paramters

```py
FRAMEWORK = 'tf'            # tf/theano
MODEL_NAME = 'Anna'         # An AI has to have a name! Also the subdirectory name
MODEL_VERSION = 1           # The version is used for the file name
VERSION_UPDATE = 1200       # New version interval, in seconds
SAVE_INTERVAL = 1000        # Save interval in iterations
# It is recommended to set this higher on non-SSD hard disks as dumping is expensive
DUMPING_INTERVAL = 5000     # Experience memory dumping interval in iterations
REPORT_INTERVAL = 1000      # Frames before reporting, if not verbose
THROTTLING_PERIOD = 2       # Frames to skip before sampling the experience log and training again
ACTIONS = 6                 # Number of valid actions
INITIAL_GAMMA = 0.6         # Low confidence in predictions while exploring
# A large gamma for a game where positive rewards are usually closely-packed can eventually cause an overflow
FINAL_GAMMA = 0.8           # High confidence when perfecting the technique
OBSERVE_PERIOD = 5000       # Frames to observe before training
EXPLORE_PERIOD = 1000000    # Iterations over which to anneal EPSILON from initial to final
INITIAL_EPSILON = 0.1       # Starting value of EPSILON
FINAL_EPSILON = 0.001       # Final value of EPSILON
MEMORY_SIZE = 50000         # Number of previous transitions to remember
BATCH = 32                  # Size of experiences to train on
FRAMES_PER_ACTION = 1       # The delay before taking another action (set to 1 for no delay)
LEARNING_RATE = 1e-4        # Our network's learning rate
FRAMES_PER_SAMPLE = 4       # How many frames to stack per sample, good for detecting time-based amounts like velocity
RESTORE_STATE = True        # Whether or not to restore a state if found
RESTORE_MEMORY = True       # Whether or not to restore the experience memory if found
# The image size, images are rotated 90 degrees in a matrix so the height is rows and the width is columns
# It's better to use a square image because, otherwise, lines might get jagged or completely disappear while resizing
IMG_ROWS , IMG_COLS = 132, 132
```

#### Important functions:

`build_model()`

This function builds our Keras model.

`save_model()`

This function saves our model (the design and weights) along with other relevant progress information like the current iteration, the model version and our epsilon/gamma paramters.

`dump_memory()`

This function dumps the experience memory (states, actions, rewards) to a file.

`load_model()`

Loads the previously learned weights for our model.

`lookup_state()`

Checks for a saved state, if found loads it.

`lookup_experience()`

Checks for a dumped memory, if found loads it.

`exercise_network()`

Starts exercising the network using the game in train/test mode.

# Disclaimer
This project was inspired by the following Keras-FlappyBird project. Initially I borrowed the same model but later modified it to better suite Pacman. You can read more about it here:
https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

I do not own the game implementation used, you can find it in the following repo:
https://github.com/greyblue9/pacman-python
