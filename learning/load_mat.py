import scipy.io as scio
import matplotlib.pyplot as plt
import tensorflow as tf

# dataFile = '/home/dl/qingyi/SVHN/test_32x32.mat'
dataFile = 'C:/Users/ZhangXuemiao/temp/train_32x32'
data = scio.loadmat(dataFile)
print(data.keys)
print(data.__len__())
keys = data.keys()
for i in keys:
    print(i)
print(data['__version__'])
print(data['__globals__'])
print(data['__header__'])
print('-----------------------------------------------------------------')
print(type(data['y']))
print(data['y'])
print(data['y'].shape)

print('------------------------------------------------------------------')
# print(data['X'])
print(type(data['X']))
print(data['X'].shape)

print('------------------------------------------------------------------')
# id = 3
# labels = data['y']
# print(labels[id])
# images = data['X']
# image = images[:, :, :, id]
# print(image.shape)
# plt.imshow(image)
# plt.show()


tfreocrd_filepath = 'C:/Users/ZhangXuemiao/temp/train.tfrecords'

for serialized_example in tf.python_io.tf_record_iterator(tfreocrd_filepath):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image_raw'].bytes_list.value
    label = example.features.feature['label'].int64_list.value

    print(label)
