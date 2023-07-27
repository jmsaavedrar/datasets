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
      
    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""    
#        cats = range(345)
#         with open('categories.txt') as f:
#             for category in f:
#                 cats.append(category.strip())
                
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({            
                'image': tfds.features.Image(shape=(None, None, 1)),
                'label': tfds.features.ClassLabel(names=range(345)),
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
            'train': self._generate_examples(os.path.join(self.path,'train.txt')),
            'test_known' : self._generate_examples(os.path.join(self.path,'test_known.txt')),
            'test_unknown' : self._generate_examples(os.path.join(self.path,'test_unknown.txt'))
            }
    
    def _generate_examples(self, fname):
        """Yields examples."""
        # TODO(tdfs_mnist): Yields (key, example) tuples from the dataset
        with open(fname) as flist :
            for i , f in enumerate(flist):
                print(i, f)            
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
          
