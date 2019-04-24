import datetime
import json
import os
import pprint
import random
import string
import sys
import tensorflow as tf

assert 'COLAB_TPU_ADDR' in os.environ, 'ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!'
TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
print('TPU address is', TPU_ADDRESS)

from google.colab import auth
auth.authenticate_user()
with tf.Session(TPU_ADDRESS) as session:
  print('TPU devices:')
  pprint.pprint(session.list_devices())

  # Upload credentials to TPU.
  with open('/content/adc.json', 'r') as f:
    auth_info = json.load(f)
  tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
  # Now credentials are set for all future sessions on this TPU.

BUCKET  = 'mongolian-bert-models'
DATASET = 'eduge'
LEN     = 512
OUTPUT_DIR = 'gs://{}/{}/{}_cased'.format(BUCKET, DATASET, LEN)
print(OUTPUT_DIR)
tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

sys.path.append("./mongolian-bert/bert")
import modeling
import optimization
import run_classifier
import tokenization

VOCAB_FILE      = os.path.join("./mongolian-bert/{}".format(MODEL), 'mn_cased.vocab')
CONFIG_FILE     = os.path.join("./mongolian-bert/{}".format(MODEL), 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, MODEL_CHECKPOINT)
DO_LOWER_CASE   = False


