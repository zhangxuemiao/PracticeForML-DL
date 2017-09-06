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


def encode_to_tfrecords(image_filepth_re,data_root_dir, filename='data.tfrecords'):
    writer = tf.python_io.TFRecordWriter(data_root_dir + '/' + filename)
    num_example = 0;
    fs = glob.iglob(image_filepth_re)
    for image_path in fs:
        captcha_text = parse_captcha_text(image_path)
        label_vector = text2vec(captcha_text)

        image  = Image.open(image_path)
        image = np.array(image)

        height, width, nchannel = image.shape

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'nchannel': tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
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
        'nchannel': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, tf.stack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32),
    ]), name='image')

    label = tf.decode_raw(example['label'], tf.int64)
    label = tf.reshape(label, tf.stack([4 * CHAR_SET_LEN]), name='label')

    print(image,label)
    return image, label


def get_batch(image, label, batch_size, crop_size):
    # 数据扩充变换
    print("batch_size:",batch_size)
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3]) # 随机剪裁
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化

    print("distorted_image->",distorted_image)

    # 生成batch
    """
    shuffle_batch函数的参数：capacity用于定义shuffle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该
                            足够大，保证数据打的足够乱
    """
    '''
    shuffle_batch()但需要注意的是它是一种图运算，要跑在sess.run()里
    '''
    images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size, num_threads=8, capacity=20000, min_after_dequeue=5000)
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


def main():
    train_filepth_re = 'C:/tmp/captcha/image/train/*.jpg'
    test_filepth_re = 'C:/tmp/captcha/image/test/*.jpg'
    data_root_dir = 'C:/tmp/captcha/image/TFRecords/'
    # test(train_filepth_re)
    # encode_to_tfrecords(train_filepth_re, data_root_dir)
    # encode_to_tfrecords(test_filepth_re, data_root_dir, filename='testdata.tfrecords')
    # parse_captcha_text('C:/tmp/captcha/image/test/1#ot4G.jpg')

    image, label = decode_from_tfrecords(data_root_dir + 'data.tfrecords')
    batch_image, batch_label = get_batch(image, label, 128, 60)
    '''
    自思：
        虽然只显示的执行了一次get_batch，事实上，这一次“现实的执行”也没有真正意义上的执行，
        这行代码的作用不是要真正意义上获取一批数据(value)，而是在构建Graph，增加ops，
        因为get_batch方法得到的是tensor，而不是真正的数据

        从而就可以很好的理解为什么存在每次迭代(iteration)都会执行一次get_batch操作了：
        session之前，graph已经定义好了，然后每次迭代，就会执行一次session.run,
        二每一次run都会将graph中相应的操作(operation)执行一遍，如get_batch这个operation，
        也就是，迭代了多少次，就会执行多少次session.run,也就相应的执行多少次get_batch这个operation

        正如这句教程：shuffle_batch()但需要注意的是它是一种图运算，要跑在sess.run()里
    '''



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
                text = vec2text(label_np)
                # print(text)
                showImage(text, image_np)


                batch_image_np, batch_label_np = session.run([batch_image, batch_label])
                print(batch_image_np.shape, batch_label_np.shape)
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop() # queue需要关闭，否则报错
        coord.join(threads)


if __name__ == '__main__':
    main()


'''
经验总结：
        要养成为每个operation增加name，如：placeholder，Variable，以及一些其他图操作(tf.multiply, add, shuffle_batch, reshape等)；
        为每个tensor增加name；调试的时候非常有用
'''
