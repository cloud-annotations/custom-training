export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

BASE_PATH=/Users/niko/Desktop/custom-training
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=${BASE_PATH}/output/pipeline.config
TRAINED_CKPT_PREFIX=${BASE_PATH}/output/checkpoint/model.ckpt-3000
EXPORT_DIR=object_detection/exported_model
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}