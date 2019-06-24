import os
import io
import json
import random

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def generate_tf_record(annotations, label_map_path, output):
  # Create a train.record TFRecord file.
  with tf.python_io.TFRecordWriter(output) as writer:
    # Load the label map we created.
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    # Get a list of all images in our dataset.
    image_names = [image for image in annotations.keys()]

    # Loop through all the training examples.
    for idx, image_name in enumerate(image_names):
      # Make sure the image is actually a file
      img_path = os.path.join(os.environ['DATA_DIR'], image_name)    
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

      # The class text is the label name and the class is the id. If there are 3
      # cats in the image and 1 dog, it may look something like this:
      # classes_text = ['Cat', 'Cat', 'Dog', 'Cat']
      # classes      = [  1  ,   1  ,   2  ,   1  ]

      # For each image, loop through all the annotations and append their values.
      for annotation in annotations[image_name]:
        xmins.append(annotation['x'])
        xmaxs.append(annotation['x2'])
        ymins.append(annotation['y'])
        ymaxs.append(annotation['y2'])
        label = annotation['label']
        classes_text.append(label.encode('utf8'))
        classes.append(label_map_dict[label])
      
      # Create the TFExample.
      try:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(image_name.encode('utf8')),
          'image/source_id': dataset_util.bytes_feature(image_name.encode('utf8')),
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
          # Write the TFExample to the TFRecord.
          writer.write(tf_example.SerializeToString())
      except ValueError:
        print('Invalid example, ignoring.')