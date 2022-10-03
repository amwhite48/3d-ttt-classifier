"""shape_dataset dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import numpy as np
import random

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
"""


class ShapeDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for shape_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'voxels': tfds.features.Tensor(shape=(16, 16, 16), dtype=tf.float64),
            'label': tfds.features.ClassLabel(names=['teapot', 'table', 'cyl']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    p = Path('..')
    return {
        'train': self._generate_examples(p),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(shape_dataset): Yields (key, example) tuples from the dataset
    for f in path.glob('*.npy'):
      image_id = random.getrandbits(256)
      if f.name.startswith('teapot'):
        label = 'teapot'
      elif f.name.startswith('table'):
        label = 'table'
      else:
        label = 'cyl'
      yield image_id, {
          'voxels': np.load(f),
          'label' : label
      }
    
