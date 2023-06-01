import tensorflow_datasets as tfds
import tfds_mnist.tfds_mnist as tfds_mnist

if __name__ == '__main__' :
    dir ='~/tensorflow_datasets/'
    ds = tfds.load('tfds_mnist',data_dir = dir)
    ds_test = ds['test'].shuffle(1024).batch(10)
        
    ds_test = ds_test.take(1)
    for sample in ds_test:
        print(sample['image'].numpy())
        