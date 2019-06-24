from object_detection.utils import config_util

def override_pipeline(override_dict, pipeline):
  configs = config_util.get_configs_from_pipeline_file(pipeline, config_override=override_dict)
  pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
  config_util.save_pipeline_config(pipeline_config, pipeline)