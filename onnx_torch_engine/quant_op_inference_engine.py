# from __future__ import absolute_import
import torch
import torch.nn as nn
import copy
from .util import*
from torch.autograd import Variable

class quanterOpInferenceEngine(object):
    def __init__(self,cfg):
        super(quanterOpInferenceEngine,self).__init__()
        self.cfg=cfg
        self.activation_inference_mode=cfg["activation_inference_mode"]
        self.activation_inference_mode_choice=["look_up_table","formular"]
        if self.activation_inference_mode not in self.activation_inference_mode_choice:
            print("Unsupport activation_inference_mode:",self.activation_inference_mode,"please choose:",self.activation_inference_mode_choice)
            assert(0)

    def q_relu(self,input_tensor,zero_point):
        if self.activation_inference_mode=="formular":
            output=torch.clamp(input_tensor,min=zero_point)
            return output
