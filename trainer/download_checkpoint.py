download_base = 'http://download.tensorflow.org/models/object_detection/'
model_file = 'faster_rcnn_resnet101_coco_2018_01_28.tar.gz'

# Download the checkpoint
opener = urllib.request.URLopener()
opener.retrieve(download_base + model_file, model_file)

# Extract all the `model.ckpt` files.
with tarfile.open(tar_path) as tar:
  for member in tar.getmembers():
    member.name = os.path.basename(member.name)
    if 'model.ckpt' in member.name:
      tar.extract(member, path='checkpoint')