
# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import tensorflow as tf

TB_DIR = './Graph'
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

def model_fn(features, labels, mode, params):
    tf.summary.image('image', features)

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    L = 4  # first convolutional layer output depth
    M = 8  # second convolutional layer output depth
    N = 12  # third convolutional layer
    O = 200  # fully connected layer

    weights = {
        "W1" : tf.Variable(tf.truncated_normal([5, 5, 1, L], stddev=0.1)),  # 5x5 patch, 1 input channel, K output channels
        "W2" : tf.Variable(tf.truncated_normal([5, 5, L, M], stddev=0.1)),
        "W3" : tf.Variable(tf.truncated_normal([4, 4, M, N], stddev=0.1)),
        "W4" : tf.Variable(tf.truncated_normal([7 * 7 * N, O], stddev=0.1)),
        "W5" : tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
    }
    biases = {
        "B1" : tf.Variable(tf.ones([L])/10),
        "B2" : tf.Variable(tf.ones([M])/10),
        "B3" : tf.Variable(tf.ones([N])/10),
        "B4" : tf.Variable(tf.ones([O])/10),
        "B5" : tf.Variable(tf.ones([10])/10)
    }   

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(features, weights["W1"], strides=[1, stride, stride, 1], padding='SAME') + biases["B1"])
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, weights["W2"], strides=[1, stride, stride, 1], padding='SAME') + biases["B2"])
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, weights["W3"], strides=[1, stride, stride, 1], padding='SAME') + biases["B3"])

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * N])

    Y4 = tf.nn.relu(tf.matmul(YY, weights["W4"]) + biases["B4"])
    Ylogits = tf.matmul(Y4, weights["W5"]) + biases["B5"]
    Y = tf.nn.softmax(Ylogits)
    
    for k, w in weights.items():
        tf.summary.histogram(k, w)
    for k, b in biases.items():
        tf.summary.histogram(k, b)
        
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
        # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical instability
        # problems with log(0) which is NaN
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=tf.one_hot(labels, 10))
        cross_entropy = tf.reduce_mean(cross_entropy)*100
        # % of correct answers found in batch
        predictions = tf.argmax(Y,1)
        accuracy = tf.metrics.accuracy(predictions, labels)

        evalmetrics = {"accuracy/mnist": accuracy}
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar("accuracy/mnist", accuracy[1])
            # the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
            lr = 0.0001 +  tf.train.exponential_decay(params["learning_rate"], tf.train.get_global_step(), 2000, 1/math.e)
            tf.summary.scalar("learning_rate", lr)
            optimizer = tf.train.AdamOptimizer(lr)
            train_step = optimizer.minimize(cross_entropy,
                                            global_step=tf.train.get_global_step())
        else:
            train_step = None
    else:
        cross_entropy = None
        train_step = None
        evalmetrics = None

    return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"classid": predictions},
            loss=cross_entropy,
            train_op=train_step,
            eval_metric_ops=evalmetrics)


def make_input_fn(batch_size, mode, shuffle=False):
    train, test = tf.keras.datasets.mnist.load_data()
    if mode == tf.estimator.ModeKeys.TRAIN:
        mnist_x, mnist_y = train
    else:
        mnist_x, mnist_y = test
    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices((
            tf.cast(mnist_x, tf.float32)/256.0, 
            mnist_y))
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
        else:
            num_epochs = 1 # end-of-input after this
        if shuffle or mode == tf.estimator.ModeKeys.TRAIN:
            ds = ds.shuffle(5000).repeat(num_epochs).batch(batch_size)
        else:
            ds = ds.repeat(num_epochs)
        ds = ds.prefetch(2)
        return ds
    return _input_fn


def train_and_evaluate(output_dir, hparams):
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        params = hparams,
        config= tf.estimator.RunConfig(
            save_checkpoints_steps = 1000,
            log_step_count_steps=500
            ),
        model_dir = output_dir
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn = make_input_fn(
            hparams['batch_size'],
            mode = tf.estimator.ModeKeys.TRAIN,
        ),
        max_steps = (50000//hparams["batch_size"]) * 20
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn = make_input_fn(
            hparams['batch_size'],
            mode = tf.estimator.ModeKeys.EVAL
        ),
        steps = 10000//hparams["batch_size"],
        start_delay_secs = 1,
        throttle_secs = 1
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__=='__main__':
    params = {
        "batch_size": 100,
        "learning_rate": 0.003,
        "pkeep": 0.75
    }
    model_name = 'mnist_3.0'
    train_and_evaluate(os.path.join(TB_DIR, model_name), params)
