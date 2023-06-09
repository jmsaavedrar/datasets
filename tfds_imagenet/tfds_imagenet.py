# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
#import datasets
from .classes import IMAGENET2012_CLASSES
import tensorflow_datasets as tfds
import glob
import skimage.io as io
import skimage.color as color
import numpy as np

_CITATION = """\
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
"""

_HOMEPAGE = "https://image-net.org/index.php"
_DATA_DIR = "/data/extracted_imagenet"
_DESCRIPTION = """\
ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, ImageNet hopes to offer tens of millions of cleanly sorted images for most of the concepts in the WordNet hierarchy. ImageNet 2012 is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images
"""

_DATA = {
    'train': glob.glob(os.path.join(_DATA_DIR, 'train_images/*.JPEG')),
    'test': glob.glob(os.path.join(_DATA_DIR, 'test_images/*.JPEG')),
    'val': glob.glob(os.path.join(_DATA_DIR, 'val_images/*.JPEG')),    
}


class Imagenet1k(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        assert len(IMAGENET2012_CLASSES) == 1000
        return self.dataset_info_from_configs(
                                              features=tfds.features.FeaturesDict({
                                                                                   'image': tfds.features.Image(shape=(None, None, 3)),
                                                                                   'label': tfds.features.ClassLabel(names=range(1000))
                                                                                   }), 
                                              supervised_keys=('image', 'label'))
        

#         return datasets.DatasetInfo(
#             description=_DESCRIPTION,
#             features=datasets.Features(
#                 {
#                     "image": datasets.Image(),
#                     "label": datasets.ClassLabel(names=list(IMAGENET2012_CLASSES.values())),
#                 }
#             ),
#             homepage=_HOMEPAGE,
#             citation=_CITATION,
#             task_templates=[ImageClassification(image_column="image", label_column="label")],
#         )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""                
        return {
            'train': self._generate_examples(image_files = _DATA["train"], split = 'train'),
            'test' : self._generate_examples(image_files = _DATA["test"], split = 'test'),
            'val' : self._generate_examples(image_files = _DATA["val"], split = 'val'),
            }

#         return [
#             datasets.SplitGenerator(
#                 name=datasets.Split.TRAIN,
#                 gen_kwargs={
#                     "archives": [dl_manager.iter_archive(os.path.join(DATA_DIR, archive)) for archive in archives["train"]],
#                     "split": "train",
#                 },
#             ),
#             datasets.SplitGenerator(
#                 name=datasets.Split.VALIDATION,
#                 gen_kwargs={
#                     "archives": [dl_manager.iter_archive(os.path.join(DATA_DIR, archive)) for archive in archives["val"]],
#                     "split": "validation",
#                 },
#             ),
#             datasets.SplitGenerator(
#                 name=datasets.Split.TEST,
#                 gen_kwargs={
#                     "archives": [dl_manager.iter_archive(os.path.join(DATA_DIR, archive)) for archive in archives["test"]],
#                     "split": "test",
#                 },
#             ),
#         ]
    
    def check_image(self, image):
        if len(image.shape) == 2 :
            image = color.gray2rgb(image)
        if image.dtype() == np.float32 :
            image = (image * 255).asptype(np.uint8)
        assert image.dtype == np.uint8, 'incorrect dtype'
        return image
         
        
    def _generate_examples(self, image_files, split):
        """Yields examples."""
        idx = 0
        for image_path in image_files:
            print(image_path, )
            #if image_path.endswith(".JPEG"):
            if split != 'test':
                # image filepath format: <IMAGE_FILENAME>_<SYNSET_ID>.JPEG                
                root, _ = os.path.splitext(image_path)                    
                _, synset_id = os.path.basename(root).rsplit("_", 1)
                label = list(IMAGENET2012_CLASSES.keys()).index(synset_id)
            else:
                label = -1
            ex = {
                  'image': self.check(io.imread(image_path)),
                  'label': label
                  }
            yield idx, ex
            idx += 1

# 
# 
# def _generate_examples(self, fname):
#         """Yields examples."""
#         # TODO(tdfs_mnist): Yields (key, example) tuples from the dataset
#         with open(fname) as flist :
#             for i , f in enumerate(flist):
#                 print(i, f)            
#                 data = f.strip().split('\t')
#                 name = data[0].strip()
#                 fimage = os.path.join(self.path, name)
#                 label = int(data[1].strip())
#                 image = io.imread(fimage)
#                 image = np.expand_dims(image, -1)
#                 yield name, {
#                     'image': image,
#                     'label': label,
#                 }
#           