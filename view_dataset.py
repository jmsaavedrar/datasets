import tensorflow_datasets as tfds
import tfds_qd.tfds_qd as tfds_qd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def view(ds, n_rows, n_cols) :    
    _, ax = plt.subplots(n_rows, n_cols)
    for i in range(n_rows) :
        for j in range(n_cols) :
            ax[i,j].set_axis_off()
            
    for batch in ds :                           
        for i, sample in enumerate(batch):                    
            sketch = sample.numpy().astype(np.uint8)
            ax[i // n_cols][i % n_cols].imshow(sketch, cmap  = 'gray')
            min_v=np.min(sketch)
            max_v=np.max(sketch)
            print('minmax {} {}'.format(min_v, max_v))
            
        plt.waitforbuttonpress(1)
    plt.show()


def map_func(image):
    image = image['image']    
    return image

if __name__ == '__main__' :
    dir ='~/tensorflow_datasets/'
    ds = tfds.load('tfds_qd',data_dir = dir)
    ds_test = ds['test'].map(map_func).shuffle(1024).batch(25)        
    #ds_test = ds_test.take(10)
    view(ds_test, 5, 5)