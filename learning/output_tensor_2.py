import tensorflow as tf
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

print(a)

with tf.Session():
  # We can also use 'c.eval()' here.
  print(c.eval())
  print(a.eval())