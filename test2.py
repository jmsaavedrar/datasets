import tensorflow_datasets as tfds
import tfds_qd.tfds_qd as tfds_qd
import tensorflow as tf
if __name__ == '__main__' :
    dir ='~/tensorflow_datasets/'
    ds = tfds.load('tfds_qd',data_dir = dir)
    ds_test = ds['test'].shuffle(1024).batch(10)
        
    ds_test = ds_test.take(1)
    for sample in ds_test:
        print(tf.shape(sample['image']))
        