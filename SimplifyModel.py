import sys
import os
import openvino as ov
#from notebook_utils import device_widget
from pathlib import Path

sys.path.append('/root/coastcao/HelloTorch/TTS')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from TTS.tts.models.vits import Vits
from TTS.config import load_config
#import torch

config_file_path = '/data/workspace/coastcao/HelloTorch/launch/python/models/config.json'
input_model_path = '/data/workspace/coastcao/HelloTorch/launch/python/models/best_model.pth'

config = load_config(config_file_path)
vits_model = Vits.init_from_config(config).cuda()
vits_model.load_checkpoint(config=None, checkpoint_path=input_model_path)
print('loaded checkpoint')

print('model info:')
print(vits_model)

