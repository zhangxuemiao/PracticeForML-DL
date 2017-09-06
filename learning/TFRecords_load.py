import tensorflow as tf
import argparse

tfreocrd_filepath = 'C:/Users/ZhangXuemiao/temp/train.tfrecords'

for serialized_example in tf.python_io.tf_record_iterator(tfreocrd_filepath):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image_raw'].bytes_list.value
    label = example.features.feature['label'].int64_list.value

    print(label)