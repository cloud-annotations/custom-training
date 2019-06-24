# Using Cloud Annotations to train models from TensorFlow's Object Detection model zoo

clone the repo:
```
git clone https://github.com/tensorflow/models.git
```

There are 3 things we need to setup:
- The TensorFlow Object Detection package
- The pipeline
- The training script (which will handle:)
  - The training data
  - The pretrained model

## Setting up the TensorFlow Object Detection package
move into the research directory:
```
cd models/research/
```

Set up the packages:
```
python setup.py sdist
(cd slim && python setup.py sdist)
```
This will create python packages dist/object_detection-0.1.tar.gz, slim/dist/slim-0.1.tar.gz

## Choosing a model type
Before moving forward we need to decide on a model type. You can find all available model types in the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

There 2 main base:
- `ssd` used by default by the Cloud Annotations tool, is great for devices that don't have a lot of power. You can detect objects very fast on devices like phones and raspberry pis.
- `faster rcnn` is good for high accuracy predictions, but runs much slower. 

In the model zoo you can find a chart with comparison of the speed and accuracy of each type.
The time is in milliseconds, but it's important to note that it is the speed of the model on a `Nvidia GeForce GTX TITAN X` gpu.
The accuracy is measured in `mAP` the higher the better.

The default model that cacli trains is the `ssd mobilenet v1` model which has the following metrics:

| Speed (ms) | mAP |
| :--------: | :-: |
|     30     |  21 |

> **Note:** As a frame of reference, I get about 15fps on my MacBook Pro, ~66ms.

In this walkthrough I'll be using the `faster r-cnn resnet101` model which has the following metrics:

| Speed (ms) | mAP |
| :--------: | :-: |
|    106     |  32 |

> **Note:** I'm guessing ~233ms on a Mac or 4fps.

Feel free to use any other model type, but make sure that the output is `Boxes` **NOT** `Masks`.

## Setting up the pipeline
Once we have decided on a model structure, we can find one of the pipeline configs provided by TensorFlow that corresponds to our model type from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).
Since we are using `faster r-cnn resnet101` we can download [`faster_rcnn_resnet101_coco.config`](https://github.com/tensorflow/models/raw/master/research/object_detection/samples/configs/faster_rcnn_resnet101_coco.config).
The pipeline config tells the TensorFlow Object Detection API what type of model we want to train, how to train it and with what data. We should be fine with the majority of the defaults, but feel free to tinker with any of the model/training params. You can get more info on the format of the config files [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).

> **Note:** There are a few things that will be dynamically changed with a script we write so that the config will always work with our data. These include: `num_classes`, `fine_tune_checkpoint`, `label_map_path` and `input_path`.

## Setting up the training script
When we start a training run our object storage bucket gets mounted to the training service. This gives us access to all of our images and an `_annotations.json` file with all of our bounding box annotations. We can access this data via an environment variable named `DATA_DIR`.
When the training run begins it looks for and runs a file named `start.sh`. This is where we can prepare our data and then run the training command.

### Preparing the training data
The TensorFlow Object Detection API expects our data to be in the format of TFRecord so we will need to write and run a conversion script.

> **Note:** If you don't care about how this all works, feel free to download these scripts from my [GitHub](TODO).

The format of the `_annotations.json` looks something like this:
```
{
  "version": "1.0",
  "type": "localization",
  "labels": ["Cat", "Dog"],
  "annotations": {
    "image1.jpg": [
      {
        "x": 0.7255949630314233,
        "x2": 0.9695875693160814,
        "y": 0.5820120073891626,
        "y2": 1,
        "label": "Cat"
      },
      {
        "x": 0.8845598428835489,
        "x2": 1,
        "y": 0.1829972290640394,
        "y2": 0.966248460591133,
        "label": "Dog"
      }
    ]
  }
}
```

Along with the TFRecord we also need a label map protobuf. The label map is what maps an integer id to a text label name. The ids are indexed by 1, meaning the first label will have an id of 1 not 0.
This is an example of what a label map for our `_annotations.json` example would look like:
```
item {
  id: 1
  name: 'Cat'
}

item {
  id: 2
  name: 'Dog'
}
```

The TFRecord format is a collection of serialized feature dicts, each looking something like this:
```
{
  'image/height': 1800,
  'image/width': 2400,
  'image/filename': 'image1.jpg',
  'image/source_id': 'image1.jpg',
  'image/encoded': ACTUAL_ENCODED_IMAGE_DATA_AS_BYTES,
  'image/format': 'jpeg',
  'image/object/bbox/xmin': [0.7255949630314233, 0.8845598428835489],
  'image/object/bbox/xmax': [0.9695875693160814, 1.0000000000000000],
  'image/object/bbox/ymin': [0.5820120073891626, 0.1829972290640394],
  'image/object/bbox/ymax': [1.0000000000000000, 0.9662484605911330],
  'image/object/class/text': (['Cat', 'Dog']),
  'image/object/class/label': ([1, 2])
}
```

First things first, let's generate a label map! Let's make a file named `generate_label_map.py`.
```python
import os
import json

# Open _annotations.json, os.environ['DATA_DIR'] is the directory where all of 
# our bucket data is stored.
with open(os.path.join(os.environ['DATA_DIR'], '_annotations.json')) as f:
  annotations = json.load(f)['annotations']

# Loop through each image and through each image's annotations and collect all
# the labels into a set. We could also just use labels array, but this could
# include labels that aren't used in the dataset.
labels = list({a['label'] for image in annotations.values() for a in image})

# Create a file named label_map.pbtxt
with open('label_map.pbtxt', 'w') as file:
  # Loop through all of the labels and write each label to the file with an id. 
  for idx, label in enumerate(labels):
    file.write('item {\n')
    file.write('\tname: \'{}\'\n'.format(label))
    file.write('\tid: {}\n'.format(idx + 1)) # indexes must start at 1.
    file.write('}\n')
```

Now that we have our label map, we can build our TFRecord. Let's make another file named `generate_tf_record.py`.
```python
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
```

> **Note:** There are a few extra things that we can do here, like shuffling the data and splitting it into training and validation sets.
> We can also shard the TFRecord if we have a few thousand images. To learn more check out the docs [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md).

### Downloading a pretrained model checkpoint
Training a model from scratch can take days and tons of data. We can mitigate this by using a pretrained model checkpoint.
Instead of starting from nothing, we can add to what was already learned with our own data.

We can get a download a checkpoint from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 

We'll write a script called `download_checkpoint.py` that will download the checkpoint to our training run.

```python
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
```

> **Note:** This script is downloading the `faster r-cnn resnet101` model, make sure you download the model type you are training.

### Inject pipeline with proper values

```python
from object_detection.utils import config_util

with open(os.path.join(os.environ['DATA_DIR'], '_annotations.json')) as f:
  annotations = json.load(f)['annotations']

num_classes = len({a['label'] for image in annotations.values() for a in image})

pipeline = 'faster_rcnn_resnet101_coco.config'

override_dict = {
  'num_classes': num_classes,
  'train_input_path': 'train.record',
  'fine_tune_checkpoint': 'checkpoint/model.ckpt',
  'label_map_path': 'label_map.pbtxt'
}

configs = config_util.get_configs_from_pipeline_file(pipeline, config_override=override_dict)
pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
config_util.save_pipeline_config(pipeline_config, pipeline)
```