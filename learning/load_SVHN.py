import argparse
import os
import sys
import tensorflow as tf
from scipy.io import loadmat

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('directory', 'C:/Users/ZhangXuemiao/temp/', 'file_dir')


def data_set(data_dir, name, num_sample_size=10000):
    filename = os.path.join(data_dir, name + '_32x32.mat')
    print(filename)
    if not os.path.isfile(filename):
        raise ValueError('Please supply a the file')
        # filename = os.path.join(data_dir,"train_32x32.mat")
    datadict = loadmat(filename)
    train_x = datadict['X']
    train_x = train_x.transpose((3, 0, 1, 2))

    print(train_x.shape)

    train_y = datadict['y'].flatten()

    print(train_y.shape)
    print(train_y)
    for i in train_y:
        if i==10:
            print("has "+str(i))
            break

    train_y[train_y == 10] = 0
    train_x = train_x[:num_sample_size]
    train_y = train_y[:num_sample_size]

    print(train_x.shape)
    print(train_y.shape)

    return train_x, train_y


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecords(images, labels, fileName):
    num_examples, rows, cols, depth = images.shape

    print('Writing', fileName)
    writer = tf.python_io.TFRecordWriter(fileName)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def split_dataset(train_x, train_y, validation_size):
    return (train_x[:-validation_size],
            train_y[:-validation_size],
            train_x[-validation_size:],
            train_y[-validation_size:])


def main(unused_argv):
    train_x, train_y = data_set(FLAGS.directory, 'train')
    test_x, test_y = data_set(FLAGS.directory, 'test')
    train_x, train_y, valid_x, valid_y = split_dataset(
        train_x, train_y, FLAGS.validation_size)

    trainFileName = os.path.join(FLAGS.directory, 'train.tfrecords')
    validationFileName = os.path.join(FLAGS.directory, 'validation.tfrecords')
    testFileName = os.path.join(FLAGS.directory, 'test.tfrecords')

    # Convert to Examples and write the result to TFRecords.
    convert_to_tfrecords(train_x, train_y, trainFileName)
    convert_to_tfrecords(test_x, test_y, validationFileName)
    convert_to_tfrecords(valid_x, valid_y, testFileName)

    print('over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='C:/Users/ZhangXuemiao/temp/',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--validation_size',
        type=int,
        default=5000,
        help="""\
          Number of examples to separate from the training data for the validation
          set.\
          """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)