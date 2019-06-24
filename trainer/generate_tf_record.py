import os
import json
import random

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

with open(os.path.join(os.environ['DATA_DIR'], '_annotations.json')) as f:
  annotations = json.load(f)['annotations']

image_files = [image for image in annotations.keys()]
label_map_dict = label_map_util.get_label_map_dict('label_map.pbtxt')
output_path = 'train.record'

def create_tf_record(examples, label_map_dict, output_filename):
  # Create a writer.
  writer = tf.python_io.TFRecordWriter(output_filename)

  # Loop through all the training examples.
  for idx, example in enumerate(examples):
    # Make sure the image is actually a file
    img_path = os.path.join(os.environ['DATA_DIR'], example)    
    if not os.path.isfile(img_path):
      continue

    # Read in the image.
    with tf.gfile.GFile(img_path, 'rb') as fid:
      encoded_jpg = fid.read()

    # Open the image with PIL so we can check that it's a jpeg and get the image
    # dimensions.
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')

    width, height = image.size

    # Initialize all the arrays.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for annotation in annotations[example]:
      xmins.append(annotation['x'])
      xmaxs.append(annotation['x2'])
      ymins.append(annotation['y'])
      ymaxs.append(annotation['y2'])
      classes_text.append(annotation['label'].encode('utf8'))
      label_map_dict[annotation['label']]
      classes.append(label_map_dict[annotation['label']])
    
    try:
      tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(example.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(example.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
      }))
      if tf_example:
        writer.write(tf_example.SerializeToString())
    except ValueError:
      print('Invalid example, ignoring.')

  # Close the writer.
  writer.close()

# Create the records.
create_tf_record(image_files, label_map_dict, output_path)