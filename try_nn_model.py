import torch.nn as nn
import torch
#TODO:下次学 model 的具体事情，包括 梯度等细节。
class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()
