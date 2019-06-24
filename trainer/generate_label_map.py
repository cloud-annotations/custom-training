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