import tensorflow_datasets as tfds
import tfds_mnist.tfds_mnist
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    ds = tfds.load('tfds_mnist')
    ds_train = ds['train']
    ds_train = ds_train.batch(10)
    print(len(ds_train))
    for sample in ds_train:
        print(tf.shape(sample['image']))
        #plt.imshow(sample['image'], cmap = 'gray')
        #plt.waitforbuttonpress(1)