import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

v = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='v')
v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')

global_step = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='global_step', trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99, global_step)

for ele1 in tf.trainable_variables():
    print(ele1.name)
for ele2 in tf.all_variables():
    print(ele2.name)