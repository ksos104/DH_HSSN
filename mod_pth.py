from typing import OrderedDict
import torch

model = torch.load('checkpoints/resnetv1d152_batch256_20200708-e79cb6a2.pth')
state_dict = model['state_dict']
no_backbone = OrderedDict()
new_state_dict = OrderedDict()

for key, value in state_dict.items():
    new_key = key
    if 'backbone' in new_key:
        new_key = '.'.join(new_key.split('.')[1:])
    no_backbone[new_key] = value

for key, value in no_backbone.items():
    new_key = key
    if 'stem' in new_key:
        new_key = new_key.replace('0.conv', '0')
        new_key = new_key.replace('0.bn', '1')
        new_key = new_key.replace('1.conv', '3')
        new_key = new_key.replace('1.bn', '4')
        new_key = new_key.replace('2.conv', '6')
        new_key = new_key.replace('2.bn', '7')
    elif 'layer1.0.downsample' in new_key:
        new_key = new_key.replace('downsample.1', 'downsample.2')
        new_key = new_key.replace('downsample.0', 'downsample.1')
    new_state_dict[new_key] = value

model['state_dict'] = new_state_dict
torch.save(model, 'checkpoints/resnetv1d152_batch256_20220826.pth')