export PYTHONPATH=$PYTHONPATH:`pwd`
python TTS/bin/synthesize.py --config_path /root/coastcao/HelloTorch/launch/python/models/G_config.json --model_path /root/coastcao/HelloTorch/launch/python/models/G_model.pth  --text "Thank you for your support, looking foward to continuing our cooperation next time"
