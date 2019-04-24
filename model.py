MODEL            = 'model-32k'
MODEL_CHECKPOINT = 'model.ckpt-4000000'
MODEL_BUCKET     = 'gs://mongolian-bert-models/model-32k-512-4000000'
import os
from os.path import exists, join, basename, splitext

is_on_colab = True
project_path = 'mongolian-bert'
try:
  import colab
except ModuleNotFoundError:
  is_on_colab = False
  project_path = '../../mongolian-bert'
  
import sys
sys.path.append(project_path)

if is_on_colab:
  # we are on Colab, clone our project
  if not exists(project_path):
    print("checking out")

from tokenization_sentencepiece import FullTokenizer
model_path = join(project_path, "sentencepiece")
tokenizer  = FullTokenizer(model_file=join(model_path, 'mn_cased.model'), vocab_file=join(model_path, 'mn_cased.vocab'), do_lower_case=False)
