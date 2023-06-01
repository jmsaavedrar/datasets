import tensorflow_datasets as tfds
import tfds_mnist.tfds_mnist as tfds_mnist

if __name__ == '__main__' :
    dir ='~/tensorflow_datasets/'
    ds = tfds.load('tfds_mnist',data_dir = dir)
    ds_train = ds['train']
    for sample in ds_train :
        print(sample['image'])    