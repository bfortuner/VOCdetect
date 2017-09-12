import os
import socket
import constants as c
import competitions.pascalvoc as comp


# Main config
HOSTNAME = socket.gethostname()
PROJECT_NAME = comp.PROJECT_NAME
PROJECT_PATH = comp.PROJECT_PATH
PROJECT_TYPE = comp.PROJECT_TYPE
IMG_INPUT_FORMATS = comp.IMG_INPUT_FORMATS
IMG_TARGET_FORMATS = comp.IMG_TARGET_FORMATS
IMG_DATASET_TYPES = comp.IMG_DATASET_TYPES
METADATA_PATH = comp.METADATA_PATH #os.path.join(PROJECT_PATH, 'metadata.csv')
PATHS = comp.PATHS
LABEL_NAMES = comp.LABEL_NAMES
LABEL_TO_IDX = comp.LABEL_TO_IDX
IDX_TO_LABEL = comp.IDX_TO_LABEL

# AWS Config
AWS_ACCESS_KEY = os.getenv('KAGGLE_AWS_ACCESS_KEY', 'dummy')
AWS_SECRET_KEY = os.getenv('KAGGLE_AWS_SECRET_ACCESS_KEY', 'dummy')
AWS_REGION = 'us-west-1'
AWS_SES_REGION = 'us-west-2'
ES_ENDPOINT = 'search-kagglecarvana-s7dnklyyz6sm2zald6umybeuau.us-west-1.es.amazonaws.com'
ES_PORT = 80
KIBANA_URL = 'https://search-kagglecarvana-s7dnklyyz6sm2zald6umybeuau.us-west-1.es.amazonaws.com/_plugin/kibana'
TIMEZONE = 'US/Pacific'

# External Resources
# If True, you must setup an S3 bucket, ES Instance, and SES address
S3_ENABLED = False #bool(os.getenv('KAGGLE_S3_ENABLED', False))
ES_ENABLED = False #bool(os.getenv('KAGGLE_ES_ENABLED', False))
EMAIL_ENABLED = bool(os.getenv('KAGGLE_SES_ENABLED', False))


# Email Notifications
ADMIN_EMAIL = 'bfortuner@gmail.com'
USER_EMAIL = 'bfortuner@gmail.com'
