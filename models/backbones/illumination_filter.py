import numpy as np
import torch
import torch.nn as nn

class FogPassFilter_conv1(nn.Module):
    def __init__(self, inputsize, pretrained):
        super().__init__()
        
        self.hidden = nn.Linear(inputsize, inputsize//2)
        self.hidden2 = nn.Linear(inputsize//2, inputsize//4)
        self.output = nn.Linear(inputsize//4, 64)
        self.leakyrelu = nn.LeakyReLU()

        if pretrained is not None:
            self.load_weights(pretrained)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x

    def load_weights(self, pretrain_path):
        if pretrain_path is None:
            return
        # if os.path.exists(pretrain_path):
        #     checkpoint = torch.load(
        #         pretrain_path, map_location=lambda storage, loc: storage)
        # elif os.path.exists(os.path.join(os.environ.get('TORCH_HOME', ''), 'hub', pretrain_path)):
        #     checkpoint = torch.load(os.path.join(os.environ.get(
        #         'TORCH_HOME', ''), 'hub', pretrain_path), map_location=lambda storage, loc: storage)
        # else:
        #     checkpoint = torch.hub.load_state_dict_from_url(
        #         pretrain_path, progress=True, map_location=lambda storage, loc: storage)
        # if 'state_dict' in checkpoint.keys():
        #     state_dict = checkpoint['state_dict']
        # else:
        #     state_dict = checkpoint
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('alignment_head.'):
        #         new_k = k.replace('alignment_head.', '')
        #     else:
        #         continue  # ignore the rest
        #     new_state_dict[new_k] = v
        checkpoints = torch.load(pretrain_path)  
        self.load_state_dict(checkpoints['fogpass1_state_dict'])
    

class FogPassFilter_res1(nn.Module):
    def __init__(self, inputsize, pretrained):
        super().__init__()
        
        self.hidden = nn.Linear(inputsize, inputsize//8)
        self.output = nn.Linear(inputsize//8, 64)
        self.leakyrelu = nn.LeakyReLU()

        if pretrained is not None:
            self.load_weights(pretrained)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        
        return x

    def load_weights(self, pretrain_path):
        if pretrain_path is None:
            return
        checkpoints = torch.load(pretrain_path)  
        self.load_state_dict(checkpoints['fogpass2_state_dict'])