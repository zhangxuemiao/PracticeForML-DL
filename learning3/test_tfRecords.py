# coding: utf-8
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/captcha/images/', 'the directory stored image files and label file')


def parse_captcha_text(image_filepth):
    """
    image_filepth: 333#abc3.jpg
    """
    splits = image_filepth.split('#')
    captcha_text = splits[1].split('.')[0]
    return captcha_text


def convert2gray(img):
    if len(img.shape) > 2:
        gray_img = np.mean(img, -1)
        return gray_img
    else:
        return img
    pass


def text2vec(text):
    text_len = len(text)
    if text_len > 4:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(4 * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


def encode_to_tfrecords(image_file_path_re, data_root_dir, filename='data.tfrecords'):
    writer = tf.python_io.TFRecordWriter(data_root_dir + '/' + filename)
    num_example = 0
    fs = glob.iglob(image_file_path_re)

    for image_path in fs:
        captcha_text = parse_captcha_text(image_path)
        label_vector = text2vec(captcha_text)

        image = Image.open(image_path)
        image = np.array(image)
        image = convert2gray(image)
        # print(image)
        height, width = image.shape
        # print('height, width->', height, width)

        # break

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_vector.tobytes()]))
        }))
        serialized = example.SerializeToString()
        writer.write(serialized)
        num_example += 1
        if num_example % 1000 == 0:
            print("已有样本数据量：", num_example)
    print("最终样本数据量：", num_example)
    writer.close()


def decode_from_tfrecords(filename, num_epoch=None):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epoch, shuffle=True)
    print(filename_queue)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    example = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
    })

    image_stack = tf.stack([tf.cast(example['height'], tf.int32),tf.cast(example['width'], tf.int32)], name='image_stack')
    image = tf.decode_raw(example['image'], tf.uint8)
    print('image->', image, 'image_stack->')
    image = tf.reshape(image, [60, 160, 8], name='image')

    label = tf.decode_raw(example['label'], tf.int64)
    label = tf.reshape(label, tf.stack([4 * CHAR_SET_LEN]), name='label')

    print('image,label->', image, label)
    return image[:, :, 0], label


def get_batch(image, label, batch_size, crop_size):
    # 数据扩充变换
    # distorted_image = tf.random_crop(image, [crop_size, crop_size])  # 随机剪裁
    # distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = tf.image.random_brightness(image, max_delta=63)  # 亮度变化
    # distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化

    print("distorted_image->", distorted_image)

    # 生成batch
    """
    shuffle_batch函数的参数：capacity用于定义shuffle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该
                            足够大，保证数据打的足够乱
    """
    '''
    shuffle_batch()但需要注意的是它是一种图运算，要跑在sess.run()里
    '''
    images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size, num_threads=8,
                                                 capacity=20000, min_after_dequeue=5000)
    # 调试显示
    # tf.summary.image('images', images)
    print(images, label_batch)
    return images, tf.reshape(label_batch, [batch_size, 4 * CHAR_SET_LEN])
    pass


def showImage(text, image):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


def test():
    train_filepth_re = 'C:/tmp/captcha/image/train/*.jpg'
    test_filepth_re = 'C:/tmp/captcha/image/test/*.jpg'
    data_root_dir = 'C:/tmp/captcha/image/TFRecords_2D/'

    encode_to_tfrecords(train_filepth_re, data_root_dir)
    # encode_to_tfrecords(test_filepth_re, data_root_dir, filename='testdata.tfrecords')


def main():
    train_filepth_re = 'C:/tmp/captcha/image/train/*.jpg'
    test_filepth_re = 'C:/tmp/captcha/image/test/*.jpg'
    data_root_dir = 'C:/tmp/captcha/image/TFRecords_2D/'
    # test(train_filepth_re)
    # encode_to_tfrecords(train_filepth_re, data_root_dir)
    # encode_to_tfrecords(test_filepth_re, data_root_dir, filename='testdata.tfrecords')
    # parse_captcha_text('C:/tmp/captcha/image/test/1#ot4G.jpg')

    image, label = decode_from_tfrecords(data_root_dir + 'data.tfrecords')

    print('image, label->', image, label)

    batch_image, batch_label = get_batch(image, label, 128, 60)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        # session.run(init)
        session.run(tf.local_variables_initializer())
        # session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for l in range(10):
                '''
                每run一次，就会指向下一个样本，一直循环
                '''
                # 塞数据
                '''
                自思：
                在ssession.run之前，graph已经建好，各个node、tensor都已经定义好，但是却没有数据流通，都是静态的
                二session.run就是要使真个graph流动起来，往tensor中不断地输入数据
                '''

                '''
                image_np, label_np = session.run([image, label])
                image和label都是tensor，image_np和label_np都是tensor中的值(value)
                也就是session.run把tensor中的值取出来
                '''
                image_np, label_np = session.run([image, label])

                print('image_np.shape->',image_np.shape)

                # text = vec2text(label_np)
                # print(text)
                # showImage(text, image_np)

                batch_image_np, batch_label_np = session.run([batch_image, batch_label])

                text1 = vec2text(batch_label_np[0])
                print(text1)
                showImage(text1, batch_image_np[0])

                print(batch_image_np.shape, batch_label_np.shape)
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()  # queue需要关闭，否则报错
        coord.join(threads)


if __name__ == '__main__':
    main()
    # test()

'''
经验总结：
        要养成为每个operation增加name，如：placeholder，Varaiable，以及一些其他图操作(tf.multiply, add, shuffle_batch等)；
        为每个tensor增加name；调试的时候非常有用
'''
