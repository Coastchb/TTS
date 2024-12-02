import sys
import os
import openvino as ov
#from notebook_utils import device_widget
from pathlib import Path
import torch

sys.path.append('/root/coastcao/HelloTorch/TTS')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from TTS.tts.models.vits import Vits
from TTS.config import load_config
#import torch

model_dir = '/root/coastcao/HelloTorch/launch/python/models'
config_file_path = '{0}/config.json'.format(model_dir)
input_model_path = '{0}/best_model.pth'.format(model_dir)

config = load_config(config_file_path)
vits_model = Vits.init_from_config(config)
vits_model.load_checkpoint(config=None, checkpoint_path=input_model_path)
print('loaded checkpoint')

print('model info:')
print(vits_model)

if hasattr(vits_model, "module"):
    print('model has module')
    model_state = vits_model.module.state_dict()
else:
    print('model has not module')
    model_state = vits_model.state_dict()

print('model state dict:')

disc_keys = []
for key, value in model_state.items():
    print(key, value)
    if key.startswith('disc'):
        disc_keys.append(key)
        print('removed key: ', key)
print(disc_keys)

for key in disc_keys:
    model_state.pop(key)


state = {
        "config": config,
        "model": model_state,
        "epoch": 0,
    }

torch.save(state, '{0}/G_model.pth'.format(model_dir))
