import numpy as np
import time
import sys

print("## Checking Keras\n\n")
import keras.backend as K
backend = K.backend()

vlen = 10 * 30 * 768
iters = 1000
v = np.random.rand(vlen)
tv = np.exp(v)


if backend == 'theano':
    print("\n\n## Checking Theano\n\n")
    import theano
    from theano import function, config, shared, sandbox
    import theano.tensor as T

    print("Version", theano.__version__)

    print("\nTesting...")

    x = shared(np.asarray(v, config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()

    for i in range(iters):
        r = f()

    t1 = time.time()

    print("\nResult", r)
    print("True Values", tv)
    print("Looping %d times took %f seconds" % (iters, t1 - t0))

    if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Theano is using the CPU', file=sys.stderr)
    else:
        print('Theano is using the GPU')

elif backend == 'tensorflow':
    print("\n\n## Checking TensorFlow\n\n")
    import tensorflow as tf
    from tensorflow.python.client import device_lib

    print("Version", tf.__version__)

    print("\nTesting...")

    x = tf.constant(v, shape=[vlen], name='x')


    gpu = False
    print(device_lib.list_local_devices())
    for device in device_lib.list_local_devices():
        if device.device_type == 'GPU':
            gpu = True

    if gpu:
        with tf.device('/gpu:0'):
            op = tf.exp(x)
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                t0 = time.time()
                for i in range(iters):
                    print("Result", sess.run(op))
                    print("True Values", tv)

                t1 = time.time()
                print("Looping %d times took %f seconds" % (iters, t1 - t0))
                print("TensorFlow is using the GPU")
    else:
        with tf.device('/cpu:0'):
            op = tf.exp(x)
            t0 = time.time()
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                print("Result", sess.run(op))
                print("True Values", tv)
            t1 = time.time()
            print("Looping %d times took %f seconds" % (iters, t1 - t0))
            print("TensorFlow is using the CPU", file=sys.stderr)
else:  # CNTK?
    print("There is no test available for", backend)