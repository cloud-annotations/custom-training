import os
import shutil
import ibm_boto3
from botocore.client import Config

# Things to change:
BUCKET_NAME = ''
OUTPUT_LOCATION = ''
STEPS = 3000
INSTANCE_ID = ''
ENDPOINT = 'https://s3.us.cloud-object-storage.appdomain.cloud'
API_KEY = ''


credentials = {
    'ibm_auth_endpoint': 'https://iam.ng.bluemix.net/oidc/token',
    'ibm_service_instance_id': INSTANCE_ID,
    'endpoint_url': ENDPOINT,
    'ibm_api_key_id': API_KEY,
    'config': Config(signature_version='oauth')
}

bucket = ibm_boto3.resource('s3', **credentials).Bucket(BUCKET_NAME)

print('downloading checkpoints...')
if os.path.exists('output') and os.path.isdir('output'):
    shutil.rmtree('output')
os.makedirs('output')
os.makedirs('output/checkpoint')

data_path = 'checkpoint/model.ckpt-{}.data-00000-of-00001'.format(STEPS)
index_path = 'checkpoint/model.ckpt-{}.index'.format(STEPS)
meta_path = 'checkpoint/model.ckpt-{}.meta'.format(STEPS)
bucket.download_file(os.path.join(OUTPUT_LOCATION, data_path), os.path.join('output', data_path))
bucket.download_file(os.path.join(OUTPUT_LOCATION, index_path), os.path.join('output', index_path))
bucket.download_file(os.path.join(OUTPUT_LOCATION, meta_path), os.path.join('output', meta_path))
bucket.download_file(os.path.join(OUTPUT_LOCATION, 'pipeline.config'), 'output/pipeline.config')
