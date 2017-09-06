from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import tensorflow as tf

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_data_dir', '/tmp/captcha/image/train/', 'the directory stored image files and label file')
flags.DEFINE_string('test_data_dir', '/tmp/captcha/image/test/', 'the directory stored image files and label file')


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image(image):
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def wrap_gen_captcha_text_and_image(filedir,imageCaptcha, ind):
    while True:
        text, image = gen_captcha_text_and_image(imageCaptcha)
        if image.shape == (60, 160, 3):
            imageCaptcha.write(text, filedir +str(ind)+'#'+ text + '.jpg')  # 写到文件
            return


def generateMassImages(imageCaptcha,usage_purpose="train"):
    if usage_purpose == "train":
        for i in range(20000):
            wrap_gen_captcha_text_and_image(FLAGS.train_data_dir, imageCaptcha, i)
            if i%1000==0:
                print("train images - have generate "+str(i)+' images')
    if usage_purpose == "test":
        for i in range(10000):
            wrap_gen_captcha_text_and_image(FLAGS.test_data_dir,imageCaptcha, i)
            if i%1000==0:
                print("test images - have generate "+str(i)+' images')


def showImage():
    # 测试
    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()


def main():
    imageCaptcha = ImageCaptcha()
    generateMassImages(imageCaptcha,usage_purpose="train")
    generateMassImages(imageCaptcha, usage_purpose="test")
    pass


if __name__ == '__main__':
    main()