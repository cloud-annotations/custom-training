import os
import json

from generate_label_map import generate_label_map
from generate_tf_record import generate_tf_record
from download_checkpoint import download_checkpoint
from override_pipeline import override_pipeline

MODEL_CHECKPOINT = 'faster_rcnn_resnet101_coco_2018_01_28.tar.gz'
MODEL_CONFIG = 'faster_rcnn_resnet101_coco.config'

label_map_path = 'label_map.pbtxt'
tf_record_path = 'train.record'
checkpoint_path = 'checkpoint'

# Open _annotations.json, os.environ['DATA_DIR'] is the directory where all of 
# our bucket data is stored.
with open(os.path.join(os.environ['DATA_DIR'], '_annotations.json')) as f:
  annotations = json.load(f)['annotations']

# Loop through each image and through each image's annotations and collect all
# the labels into a set. We could also just use labels array, but this could
# include labels that aren't used in the dataset.
labels = list({a['label'] for image in annotations.values() for a in image})

override_dict = {
  'num_classes': len(labels),
  'train_input_path': tf_record_path,
  'fine_tune_checkpoint': os.path.join(checkpoint_path, 'model.ckpt'),
  'label_map_path': label_map_path
}

generate_label_map(labels, label_map_path)
generate_tf_record(annotations, label_map_path, tf_record_path)
download_checkpoint(MODEL_CHECKPOINT, checkpoint_path)
override_pipeline(override_dict, MODEL_CONFIG)