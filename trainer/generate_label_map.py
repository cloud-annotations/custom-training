import os
import json

def generate_label_map(labels, output):
  # Create a file named label_map.pbtxt
  with open(output, 'w') as file:
    # Loop through all of the labels and write each label to the file with an id. 
    for idx, label in enumerate(labels):
      file.write('item {\n')
      file.write('\tname: \'{}\'\n'.format(label))
      file.write('\tid: {}\n'.format(idx + 1)) # indexes must start at 1.
      file.write('}\n')