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