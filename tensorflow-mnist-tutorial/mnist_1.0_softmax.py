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
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    XX = tf.reshape(features, [-1, 784])

    # The model
    Y = tf.nn.softmax(tf.matmul(XX, W) + b)
    
    tf.summary.histogram(W.name, W)
    tf.summary.histogram(b.name, b)
        
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # loss function
        cross_entropy = -tf.reduce_sum(tf.one_hot(labels, 10) * tf.log(Y))
        # % of correct answers found in batch
        predictions = tf.argmax(Y,1)
        accuracy = tf.metrics.accuracy(predictions, labels)

        evalmetrics = {"accuracy/mnist": accuracy}
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar("accuracy/mnist", accuracy[1])
            tf.summary.scalar("learning_rate", params["learning_rate"])
            optimizer = tf.train.GradientDescentOptimizer(params["learning_rate"])
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


def make_input_fn(mnist_x, mnist_y, batch_size, mode, shuffle=False):
    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices((
            tf.cast(tf.expand_dims(mnist_x, -1), tf.float32)/256.0, 
            tf.cast(mnist_y, tf.int32)))
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
        else:
            num_epochs = 1 # end-of-input after this
        if shuffle or mode == tf.estimator.ModeKeys.TRAIN:
            ds = ds.shuffle(5000).repeat(num_epochs).batch(batch_size)
        else:
            ds = ds.repeat(num_epochs).batch(batch_size)
        ds = ds.prefetch(2)
        return ds
    return _input_fn


def train_and_evaluate(model_dir, hparams):
    
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        params = hparams,
        config= tf.estimator.RunConfig(
            save_checkpoints_steps = 1000,
            log_step_count_steps=500
            ),
        model_dir = model_dir)
    
    train, test = tf.keras.datasets.mnist.load_data()
    
    train_x, train_y = train
    train_spec = tf.estimator.TrainSpec(
        input_fn = make_input_fn(
            train_x, train_y,
            hparams['batch_size'],
            mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = (50000//hparams["batch_size"]) * 20)
    
    test_x, test_y = test
    eval_spec = tf.estimator.EvalSpec(
        input_fn = make_input_fn(
            test_x, test_y,
            hparams['batch_size'],
            mode = tf.estimator.ModeKeys.EVAL
        ),
        #steps = 10000//hparams["batch_size"],
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
    model_name = 'mnist_1.0'
    train_and_evaluate(os.path.join(TB_DIR, model_name), params)
