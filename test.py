import tensorflow_datasets as tfds
import tfds_mnist.tfds_mnist
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ds = tfds.load('tfds_mnist')
    for sample in ds['train'] :
        plt.imshow(sample['image'], cmap = 'gray')
        plt.waitforbuttonpress(1)