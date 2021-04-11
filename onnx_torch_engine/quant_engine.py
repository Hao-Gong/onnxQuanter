# from __future__ import absolute_import
import torch
import torch.nn as nn
import onnx
from typing import List, Dict, Union, Optional, Tuple, Sequence
import copy
from .util import*
from torch.autograd import Variable

class quanterEngine(object):
    def __init__(self,cfg):
        # super(quanterEngine,self).__init__()

        self.QUANT_OP_LIST=["Conv","MatMul","Add"]
        self.parseCfg(cfg)

        self.QUANT_MAX=int(np.power(2,self.quant_precsion-1)-1)
        self.QUANT_MIN=int(-np.power(2,self.quant_precsion-1))
        self.QUANT_BIT_MAX=int(np.power(2,self.quant_precsion)-1)
        self.refresh()


    def parseCfg(self,cfg):
        self.cfg=cfg
        self.quant_precsion=self.cfg["quant_precsion"]
        self.quant_method=self.cfg["quant_method"]
        self.wt_quant_mode=self.cfg["wt_quant_mode"]
        self.ft_quant_mode=self.cfg["ft_quant_mode"]
        self.quant_granularity=self.cfg["quant_granularity"]
        self.int_mode=self.cfg["int_mode"]
        self.quant_patch=self.cfg["quant_patch"]

        for patch in self.quant_patch:
            for elt in patch:
                if elt not in self.QUANT_OP_LIST:
                    self.QUANT_OP_LIST.append(elt)

        self.quant_method_choice=["minmax","kl"]
        self.quant_mode_choice=["symmetric","asymmetric"]
        self.quant_granularity_choice=["total","channel"]
        self.int_mode_choice=["unsigned","signed"]

        if self.quant_method not in self.quant_method_choice:
            print("Unsupport quant_method:",self.quant_method,"please choose:",self.quant_method_choice)
            assert(0)
        if self.wt_quant_mode not in self.quant_mode_choice:
            print("Unsupport wt_quant_mode:",self.wt_quant_mode,"please choose:",self.quant_mode_choice)
            assert(0)
        if self.ft_quant_mode not in self.quant_mode_choice:
            print("Unsupport ft_quant_mode:",self.ft_quant_mode,"please choose:",self.quant_mode_choice)
            assert(0)
        if self.quant_granularity not in self.quant_granularity_choice:
            print("Unsupport quant_granularity:",self.quant_granularity,"please choose:",self.quant_granularity_choice)
            assert(0)
        if self.int_mode not in self.int_mode_choice:
            print("Unsupport int_mode:",self.int_mode,"please choose:",self.int_mode_choice)
            assert(0)

        if self.quant_precsion>8:
            self.quant_method="minmax"

    def refresh(self):
        self.quant_observer_table={}
        self.input_observer={}
        self.calibration_count=0
        self.getHistgram_count=0

    def calculateScaleZeroPoint(self,float_min,float_max,quant_mode,quant_granularity):
        if quant_granularity=="total":
            if self.int_mode=="signed":
                if quant_mode=="symmetric":
                    abs_float_max=max(abs(float_min),abs(float_max))
                    scale=abs_float_max/self.QUANT_MAX
                    zero_point=0

                elif quant_mode=="asymmetric":
                    float_max=max(0,float_max)
                    float_min=min(0,float_min)
                    float_range=float_max-float_min
                    float_zero_range=0-float_min
                    scale=float_range/self.QUANT_BIT_MAX
                    zero_point=int(self.QUANT_MIN+float_zero_range/scale)
                    # print(float_max,float_min,(self.QUANT_MAX-zero_point)*scale,(self.QUANT_MIN-zero_point)*scale)

            elif self.int_mode=="unsigned":

                if quant_mode=="symmetric":
                    abs_float_max=max(abs(float_min),abs(float_max))
                    scale=abs_float_max/self.QUANT_MAX
                    zero_point=self.QUANT_MAX

                elif quant_mode=="asymmetric":
                    float_max=max(0,float_max)
                    float_min=min(0,float_min)
                    float_range=float_max-float_min
                    float_zero_range=0-float_min
                    scale=float_range/self.QUANT_BIT_MAX
                    zero_point=int(float_zero_range/scale)

        elif quant_granularity=="channel":
            if self.int_mode=="signed":
                if quant_mode=="symmetric":
                    float_min=abs(float_min)
                    float_max=abs(float_max)
                    abs_float_max=np.maximum(float_min,float_max)
                    print(abs_float_max.shape)
                    scale=abs_float_max/self.QUANT_MAX
                    zero_point=np.zeros(scale.shape)

                elif quant_mode=="asymmetric":
                    print("NOT SUPPORT")
                    assert(0)
                    # float_max=max(0,float_max)
                    # float_min=min(0,float_min)
                    # float_range=float_max-float_min
                    # float_zero_range=0-float_min
                    # scale=float_range/self.QUANT_BIT_MAX
                    # zero_point=int(self.QUANT_MIN+float_zero_range/scale)

            elif self.int_mode=="unsigned":
                print("NOT SUPPORT")
                assert(0)
                # if quant_mode=="symmetric":
                #     abs_float_max=max(abs(float_min),abs(float_max))
                #     scale=abs_float_max/self.QUANT_MAX
                #     zero_point=self.QUANT_MAX

                # elif quant_mode=="asymmetric":
                #     float_max=max(0,float_max)
                #     float_min=min(0,float_min)
                #     float_range=float_max-float_min
                #     float_zero_range=0-float_min
                #     scale=float_range/self.QUANT_BIT_MAX
                #     zero_point=int(float_zero_range/scale)

        return scale,zero_point