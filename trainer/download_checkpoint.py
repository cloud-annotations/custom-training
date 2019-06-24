import os
import tarfile

import six.moves.urllib as urllib

def download_checkpoint(model, output):
  download_base = 'http://download.tensorflow.org/models/object_detection/'

  # Download the checkpoint
  opener = urllib.request.URLopener()
  opener.retrieve(download_base + model, model)

  # Extract all the `model.ckpt` files.
  with tarfile.open(model) as tar:
    for member in tar.getmembers():
      member.name = os.path.basename(member.name)
      if 'model.ckpt' in member.name:
        tar.extract(member, path=output)