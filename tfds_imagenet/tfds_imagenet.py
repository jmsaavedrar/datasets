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
from datasets.tasks import ImageClassification
from .classes import IMAGENET2012_CLASSES
import tensorflow_datasets as tfds


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
_DATA_DIR = "/data/imagenet"
_DESCRIPTION = """\
ILSVRC 2012, commonly known as 'ImageNet' is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, ImageNet hopes to offer tens of millions of cleanly sorted images for most of the concepts in the WordNet hierarchy. ImageNet 2012 is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images
"""

_DATA = {
    'train': ['train_images/train_images_{i}.tar.gz'  for i in range(5)],
    'val': ['val_images/val_images.tar.gz'],
    'test': ['test_images/test_images.tar.gz']    
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
        archives = _DATA
        return {
            'train': self._generate_examples(archives = [dl_manager.iter_archive(os.path.join(_DATA_DIR, archive)) for archive in archives["train"]],
                                             split = 'train'),
            'test' : self._generate_examples(archives = [dl_manager.iter_archive(os.path.join(_DATA_DIR, archive)) for archive in archives["test"]],
                                             split = 'val'),
            'val' : self._generate_examples(archives = [dl_manager.iter_archive(os.path.join(_DATA_DIR, archive)) for archive in archives["val"]],
                                             split = 'val'),
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

    def _generate_examples(self, archives, split):
        """Yields examples."""
        idx = 0
        for archive in archives:
            for path, file in archive:
                if path.endswith(".JPEG"):
                    if split != "test":
                        # image filepath format: <IMAGE_FILENAME>_<SYNSET_ID>.JPEG
                        root, _ = os.path.splitext(path)
                        _, synset_id = os.path.basename(root).rsplit("_", 1)
                        label = IMAGENET2012_CLASSES[synset_id]
                    else:
                        label = -1
                    ex = {"image": {"path": path, "bytes": file.read()}, "label": label}
                    yield idx, ex
                    idx += 1
