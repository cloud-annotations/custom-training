#!/bin/bash

# If any of the commands fail, echoing this message will inform cacli.
trap 'echo CACLI-TRAINING-FAILED; exit' ERR

# Install any additional requirements. We must install on the user level for it 
# to work with WML. Installing with `--no-deps` ensures we don't override the
# default packages provided by WML.
pip install --user --no-deps -r requirements.txt

# Unpack the object_detection and slim packages.
tar -xvzf object_detection-0.1.tar.gz
tar -xvzf slim-0.1.tar.gz

# Move the object_detection and slim packages.
cp -rf object_detection-0.1/object_detection .
cp -rf slim-0.1 slim/

# Cleanup. (not really necessary)
rm -rf object_detection-0.1.tar.gz
rm -rf object_detection-0.1
rm -rf slim-0.1.tar.gz
rm -rf slim-0.1

# Add slim to our python path.
export PYTHONPATH=${PWD}/slim

# Run our prep scripts.
python prepare_training.py

# Start training. ($1 is the integer of training steps provided by cacli)
python -m object_detection.model_main \
  --pipeline_config_path='faster_rcnn_resnet101_coco.config' \
  --model_dir='checkpoint' \
  --num_train_steps=$1 \
  --alsologtostderr

# Tell cacli we successfully finished training.
echo 'CACLI-TRAINING-SUCCESS'