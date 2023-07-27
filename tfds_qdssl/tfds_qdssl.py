"""tdfs_mnist dataset."""

import tensorflow_datasets as tfds
import os
import skimage.io as io
import numpy as np

class TFDS_QDSSL(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for tdfs_mnist dataset."""
    
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    def get_categories(self):
        with open('categories.txt', 'r+') as fin:
            categories =  [line.strip() for line in fin]
        return categories
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""    
#        cats = range(345)
#         with open('categories.txt') as f:
#             for category in f:
#                 cats.append(category.strip())
        categories = self.get_categories()                
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({            
                'image': tfds.features.Image(shape=(None, None, 1)),
                'label': tfds.features.ClassLabel(names=categories),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
        )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""    
        self.path = '/home/DIINF/vchang/jsaavedr/datasets/quickdraw'    
        return {
            'train': self._generate_examples(os.path.join(self.path,'train_ssl.txt')),
            'test_known' : self._generate_examples(os.path.join(self.path,'test_known_ssl.txt')),
            'test_unknown' : self._generate_examples(os.path.join(self.path,'test_unknown_ssl.txt'))
            }
    
    def _generate_examples(self, fname):
        """Yields examples."""
        # TODO(tdfs_mnist): Yields (key, example) tuples from the dataset
        with open(fname) as flist :
            for i , f in enumerate(flist):       
                if (i + 1)  % 100 == 0 :
                    print('{} {}'.format(fname, i+1))                                      
                data = f.strip().split('\t')
                name = data[0].strip()
                fimage = os.path.join(self.path, name)
                label = int(data[1].strip())
                image = io.imread(fimage)
                image = np.expand_dims(image, -1)                
                yield name, {
                    'image': image,
                    'label': label,
                }
                
          
