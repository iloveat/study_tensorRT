# -*- coding: utf-8 -*-
import torch
from net_definition import Net
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


model = Net()
model.cuda()

# load saved model
model = torch.load('../model/torch_000.model')
weights = model.state_dict()
print(weights.keys())
# print(weights['fc2.bias'])





















