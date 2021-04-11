# from __future__ import absolute_import
import torch
import torch.nn as nn
import onnx
from typing import List, Dict, Union, Optional, Tuple, Sequence
import copy
from collections import Counter
from .util import*
from torch.autograd import Variable
from .converter import onnxTorchModel
from .quant_engine import quanterEngine
from .quant_op_inference_engine import quanterOpInferenceEngine
import matplotlib.pyplot as plt

EPS=0.00000001

class onnxTorchQuanter(onnxTorchModel,quanterEngine,quanterOpInferenceEngine):
    def __init__(self,onnx_model: onnx.ModelProto,cfg:dict):
        self.cfg=cfg
        self.cfg["pad_split"]=True
        if self.check_model(onnx_model):
            onnxTorchModel.__init__(self,onnx_model,self.cfg)
            quanterEngine.__init__(self,self.cfg)
            quanterOpInferenceEngine.__init__(self,self.cfg)
            
        self.float_model=onnxTorchModel(onnx_model,cfg)

    def check_model(self,onnx_model: onnx.ModelProto):
        return True

    def klHistgramObserver(self,input):
        kl_bin=8001
        if self.quant_method=="kl" and self.calibration_count>0:
            for exc_info in self.forwardExcList:
                op_type=exc_info["type"]
                output_blob=exc_info["output"]

                if op_type=="Conv":
                    next_op_type=self.getNextTypeByInput(output_blob[0])
                    if self.checkActiInPatch(next_op_type):
                        continue

                if op_type in self.QUANT_OP_LIST:
                    ft_float_min=exc_info["quant_observer"]["ft_float_min"]
                    ft_float_max=exc_info["quant_observer"]["ft_float_max"]
                    abs_max=max(abs(ft_float_min),abs(ft_float_max))
                    # ft_float_min=-abs_max
                    # ft_float_max=abs_max
                    # exc_info["quant_observer"]["ft_float_min"]=ft_float_min
                    # exc_info["quant_observer"]["ft_float_max"]=ft_float_max
                    exec("self.data_clone={output_blob}.clone().view(-1)".format(output_blob=output_blob[0]))
                    # if op_type=="Relu":
                    if self.getHistgram_count==0:
                        kl_bin_size=2*abs_max/kl_bin
                        exc_info["quant_observer"]["kl_bin"]=kl_bin
                        exc_info["quant_observer"]["kl_bin_size"]=kl_bin_size
                        kl_histgram=np.zeros(kl_bin)
                        self.data_clone=((self.data_clone+abs_max)/kl_bin_size).long()
                        self.data_clone=self.data_clone.cpu().numpy()
                        self.data_clone[np.where(self.data_clone==kl_bin)]=kl_bin-1
                        bin_counter=Counter(self.data_clone)
                        zero_bin=int(abs_max/kl_bin_size)
                        for bin_index in bin_counter.keys():
                            kl_histgram[bin_index]+=bin_counter[bin_index]
                        kl_histgram[zero_bin]=0

                        exc_info["quant_observer"]["kl_histgram"]=kl_histgram
                    else:
                        kl_bin=exc_info["quant_observer"]["kl_bin"]
                        kl_bin_size=exc_info["quant_observer"]["kl_bin_size"]
                        kl_histgram=exc_info["quant_observer"]["kl_histgram"]
                        self.data_clone=((self.data_clone+abs_max)/kl_bin_size).long()
                        self.data_clone=self.data_clone.cpu().numpy()
                        self.data_clone[np.where(self.data_clone==kl_bin)]=kl_bin-1
                        bin_counter=Counter(self.data_clone)
                        zero_bin=int(abs_max/kl_bin_size)
                        for bin_index in bin_counter.keys():
                            kl_histgram[bin_index]+=bin_counter[bin_index]
                        kl_histgram[zero_bin]=0
                        exc_info["quant_observer"]["kl_histgram"]=kl_histgram

            print("kl observer accept number:",self.getHistgram_count)
            self.getHistgram_count+=1
            
    def calculateKL(self):
        for exc_info in self.forwardExcList:
            op_type=exc_info["type"]
            output_blob=exc_info["output"]
            if op_type=="Conv":
                next_op_type=self.getNextTypeByInput(output_blob[0])
                if self.checkActiInPatch(next_op_type):
                    continue

            if op_type in self.QUANT_OP_LIST:
                ft_float_min=exc_info["quant_observer"]["ft_float_min"]
                ft_float_max=exc_info["quant_observer"]["ft_float_max"]
                abs_max=max(abs(ft_float_min),abs(ft_float_max))
                kl_bin=exc_info["quant_observer"]["kl_bin"]
                kl_bin_size=exc_info["quant_observer"]["kl_bin_size"]
                kl_histgram=exc_info["quant_observer"]["kl_histgram"]
                target_bin=self.QUANT_BIT_MAX
                kl_histgram=kl_histgram/np.sum(kl_histgram)
                # hist_center=np.argmax(kl_histgram)

                # if hist_center<target_bin:
                #     hist_center=target_bin
                # elif hist_center>(kl_bin-target_bin):
                #     hist_center-kl_bin-target_bin

                hist_center=int(abs_max/kl_bin_size)
                # hist_center=kl_bin//2
                # zero_point_bin=int(abs_max/kl_bin_size)
                best_kl_value=100000
                best_left_shrink=0
                best_right_shrink=kl_bin-1
                hist_center_to_right=kl_bin-hist_center
                # hist_center_to_max=max(hist_center_to_right,hist_center)
                for i in range(target_bin//2,hist_center):
                    left_shrink=hist_center-i
                    right_shrink=hist_center+i+1
                    # print(left_shrink,right_shrink,hist_center)
                    left_outer_sum=np.sum(kl_histgram[:left_shrink])
                    right_outer_sum=np.sum(kl_histgram[right_shrink:])
                    
                    p_hist=kl_histgram[left_shrink:right_shrink].copy()
                    hist_len=right_shrink-left_shrink
                    hist_ratio=int(float(hist_len)/float(target_bin))
                    p_hist[0]+=left_outer_sum
                    p_hist[-1]+=right_outer_sum

                    q_hist_copy=kl_histgram[left_shrink:right_shrink].copy()
                    q_hist=np.zeros(hist_len)
                    # q_hist_target_bin=np.zeros(target_bin)

                    # reflection_table=np.array(np.array(range(len(q_hist_copy)))/hist_ratio,dtype=int)
                    # reflection_table[np.where(reflection_table>target_bin-1)]=target_bin-1
                    reflection_table=np.zeros(len(q_hist_copy))
                    for i in range(target_bin):
                        start=i*hist_ratio
                        end=start+hist_ratio
                        reflection_table[start:end]=i
                    reflection_table[target_bin*hist_ratio:]=target_bin-1

                    # print(reflection_table)
                    for t_index in range(target_bin):
                        index_table=np.where(reflection_table==t_index)
                        table_grid_size=float(len(q_hist_copy[index_table]))
                        if(table_grid_size==0):
                            table_grid_size=1
                        bin_sum_tmp=np.sum(q_hist_copy[index_table])
                        q_hist[index_table]=bin_sum_tmp/table_grid_size

                    # q_hist_zeros=len(q_hist[np.where(q_hist==0)])
                    # q_hist_nonzeros=len(q_hist)-q_hist_zeros
                    # if(q_hist_nonzeros>0):
                    #     EPS_BALANCE=EPS*q_hist_zeros/q_hist_nonzeros
                    #     q_hist[np.where(q_hist!=0)]=q_hist[np.where(q_hist!=0)]-EPS_BALANCE
                    q_hist[np.where(q_hist==0)]=EPS
                    
                    # p_hist_zeros=len(p_hist[np.where(p_hist==0)])
                    # p_hist_nonzeros=len(p_hist)-p_hist_zeros
                    # if(p_hist_nonzeros>0):
                    #     EPS_BALANCE=EPS*p_hist_zeros/p_hist_nonzeros
                    #     p_hist[np.where(p_hist!=0)]=p_hist[np.where(p_hist!=0)]-EPS_BALANCE
                    p_hist[np.where(p_hist==0)]=EPS
                    kl_current=np.sum(p_hist*np.log(p_hist/q_hist))
                    # print(len(p_hist),hist_len,left_shrink,right_shrink,kl_current,np.sum(q_hist),np.sum(p_hist))
                    # continue
                    if kl_current<best_kl_value:
                        best_kl_value=kl_current
                        best_left_shrink=left_shrink
                        best_right_shrink=right_shrink
                        
                ft_float_min_new=2*abs_max/kl_bin*best_left_shrink-abs_max
                ft_float_max_new=2*abs_max/kl_bin*best_right_shrink-abs_max
                
                if(best_kl_value<0.05):
                    exc_info["quant_observer"]["ft_float_min"]=max(ft_float_min_new,exc_info["quant_observer"]["ft_float_min"])
                    exc_info["quant_observer"]["ft_float_max"]=ft_float_max_new
                    print(ft_float_min,exc_info["quant_observer"]["ft_float_min"],ft_float_max,exc_info["quant_observer"]["ft_float_max"],best_kl_value)
                else:
                    print(ft_float_min,ft_float_min,ft_float_max,ft_float_max,best_kl_value)

    def minmaxObserver(self,input):
        output=self.forward(input)
        net_input=self.onnx_model.graph.input

        self.image_input=self.onnxBlobNameTable[net_input[0].name]
        exec("self.ft_float_min=torch.min({image_input}).cpu().numpy()".format(image_input=self.image_input))
        exec("self.ft_float_max=torch.max({image_input}).cpu().numpy()".format(image_input=self.image_input))
        if self.calibration_count==0:
            self.input_observer={"quant_observer":{"ft_float_min":self.ft_float_min,"ft_float_max":self.ft_float_max},"output":[self.image_input]}
        else:
            self.ft_float_min=min(self.ft_float_min,self.input_observer["quant_observer"]["ft_float_min"])
            self.ft_float_max=max(self.ft_float_max,self.input_observer["quant_observer"]["ft_float_max"])
            self.input_observer={"quant_observer":{"ft_float_min":self.ft_float_min,"ft_float_max":self.ft_float_max},"output":[self.image_input]}

        # # print(self.ft_float_min,self.ft_float_max)
        for exc_info in self.forwardExcList:
            op_type=exc_info["type"]
            output_blob=exc_info["output"]
            if op_type in self.QUANT_OP_LIST:
                if op_type=="Conv":
                    next_op_type=self.getNextTypeByInput(output_blob[0])
                    print("observe:",op_type," next:",next_op_type)
                    if self.checkActiInPatch(next_op_type):
                        exc_info["quant_observer"]={}
                        continue
                
                exec("self.ft_float_min=torch.min({output_blob}).cpu().numpy()".format(output_blob=output_blob[0]))
                exec("self.ft_float_max=torch.max({output_blob}).cpu().numpy()".format(output_blob=output_blob[0]))
                if self.calibration_count==0:
                    exc_info["quant_observer"]={"ft_float_min":self.ft_float_min,"ft_float_max":self.ft_float_max}
                else:
                    self.ft_float_min=min(self.ft_float_min,exc_info["quant_observer"]["ft_float_min"])
                    self.ft_float_max=max(self.ft_float_max,exc_info["quant_observer"]["ft_float_max"])
                    exc_info["quant_observer"]={"ft_float_min":self.ft_float_min,"ft_float_max":self.ft_float_max}
            else:
                exc_info["quant_observer"]={}

        print("minmax observer accept number:",self.calibration_count)
        self.calibration_count+=1

    def fake_quant_forward(self,input):
        if self.convert_weights==True:
            print("Please don't convert weights to Fix")

        ft_scale=self.input_observer["quant_observer"]["ft_scale"]
        input=(input/ft_scale).long().float()*ft_scale

        net_input=self.onnx_model.graph.input
        net_output=self.onnx_model.graph.output
        if len(net_input)==1:
            exc_str="{node_input}=input".format(node_input=self.onnxBlobNameTable[net_input[0].name])
            exec(exc_str)

        for exc_info in self.forwardExcList:
            if "exec_pad" in exc_info.keys():
                exec(exc_info["exec_pad"])
            op_type=exc_info["type"]
            if op_type =="Conv":
                var_name=exc_info["var_name"]
                self.wt_scale=exc_info["quant_observer"]["wt_scale"]
                exec("{var_name}.weight.data/=self.wt_scale".format(var_name=var_name))
                exec("{var_name}.weight.data={var_name}.weight.data.long().float()".format(var_name=var_name))
                exec("{var_name}.weight.data*=self.wt_scale".format(var_name=var_name))

                if exc_info["has_bias"]:
                    self.wt_scale=exc_info["quant_observer"]["wt_scale"]
                    input_name=exc_info["input"][0]
                    self.input_ft_scale=self.quant_observer_table[input_name]["ft_scale"]
                    self.scale_mul=self.wt_scale*self.input_ft_scale
                    exec("{var_name}.bias.data/=self.scale_mul".format(var_name=var_name))
                    exec("{var_name}.bias.data={var_name}.bias.data.long().float()".format(var_name=var_name))
                    exec("{var_name}.bias.data*=self.scale_mul".format(var_name=var_name))

                exec(exc_info["exec"])

            elif op_type=="Constant":
                var_name=exc_info["var_name"]
                self.wt_scale=exc_info["quant_observer"]["wt_scale"]
                exec("{var_name}.data/=self.wt_scale".format(var_name=var_name))
                exec("{var_name}.data={var_name}.data.long().float()".format(var_name=var_name))
                exec("{var_name}.data*=self.wt_scale".format(var_name=var_name))
                exec(exc_info["exec"])

            else:
                exec(exc_info["exec"])

        if len(net_output)==1:
            exc_str="self.net_output={node_output}".format(node_output=self.onnxBlobNameTable[net_output[0].name])
            exec(exc_str)

        return self.net_output

    def float_forward(self,input):
        return self.float_model(input)

    def quant_forward(self,input):
        ft_scale=self.input_observer["quant_observer"]["ft_scale"]
        ft_zero_point=self.input_observer["quant_observer"]["ft_zero_point"]
        input=(input/ft_scale+ft_zero_point).long().float()
        net_input=self.onnx_model.graph.input
        net_output=self.onnx_model.graph.output
        if len(net_input)==1:
            exc_str="{node_input}=input".format(node_input=self.onnxBlobNameTable[net_input[0].name])
            exec(exc_str)
        
        for exc_info in self.forwardExcList:
            if "exec_pad" in exc_info.keys():
                exec(exc_info["exec_pad"])
            op_type=exc_info["type"]

            if op_type =="Conv":
                var_name=exc_info["var_name"]
                input_name=exc_info["input"][0]
                output_name=exc_info["output"][0]
                self.input_scale=self.quant_observer_table[input_name]["ft_scale"]
                self.wt_scale=exc_info["quant_observer"]["wt_scale"]
                self.ft_scale=exc_info["quant_observer"]["ft_scale"]
                self.rescale=self.input_scale*self.wt_scale/self.ft_scale
                exec(exc_info["exec"])
                exec(exc_info["quant_add_bias_exc"])
                exec("{var_name}*=self.rescale".format(var_name=output_name))
                exec("{var_name}={var_name}.long().float()".format(var_name=output_name))
                self.output_zero_point=exc_info["quant_observer"]["ft_zero_point"]
                exec("{var_name}={var_name}+self.output_zero_point".format(var_name=output_name))
                exec("torch.clamp({var_name},min=self.QUANT_MIN,max=self.QUANT_MAX)".format(var_name=output_name))
                # exec("print({var_name}.shape)".format(var_name=output_name))
                # exec("print({var_name}_quant_bias.shape)".format(var_name=output_name))
                # exc_info["quant_observer"]["ft_zero_point"]
                

            elif op_type=="MatMul":
                var_name=exc_info["var_name"]
                input_name_0=exc_info["input"][0]
                input_name_1=exc_info["input"][1]
                output_name=exc_info["output"][0]
                self.input_scale_0=self.quant_observer_table[input_name_0]["ft_scale"]
                self.input_scale_1=self.quant_observer_table[input_name_1]["ft_scale"]
                self.ft_scale=exc_info["quant_observer"]["ft_scale"]
                self.rescale=self.input_scale_0*self.input_scale_1/self.ft_scale
                exec(exc_info["exec"])
                exec(exc_info["quant_add_bias_exc"])
                exec("{var_name}*=self.rescale".format(var_name=output_name))
                exec("{var_name}={var_name}.long().float()".format(var_name=output_name))
                self.output_zero_point=exc_info["quant_observer"]["ft_zero_point"]
                exec("{var_name}={var_name}+self.output_zero_point".format(var_name=output_name))
                exec("torch.clamp({var_name},min=self.QUANT_MIN,max=self.QUANT_MAX)".format(var_name=output_name))

            elif op_type=="Constant":
                exec(exc_info["exec"])

            elif op_type=="Softmax":
                var_name=exc_info["var_name"]
                input_name=exc_info["input"][0]
                output_name=exc_info["output"][0]
                self.input_scale=self.quant_observer_table[input_name]["ft_scale"]
                self.input_zero_point=self.quant_observer_table[input_name]["ft_zero_point"]
                exec("self.input_clone={var_name}.clone()".format(var_name=input_name))
                exec("{var_name}=({var_name}-self.input_zero_point)*self.input_scale".format(var_name=input_name))
                exec(exc_info["exec"])
                exec("torch.clamp({var_name},min=self.QUANT_MIN,max=self.QUANT_MAX)".format(var_name=output_name))
                exec("{var_name}=self.input_clone.clone()".format(var_name=input_name))

            elif op_type=="Add":
                var_name=exc_info["var_name"]
                output_name=exc_info["output"][0]
                self.add_list=[]
                self.output_scale=self.quant_observer_table[output_name]["ft_scale"]
                self.output_zero_point=self.quant_observer_table[output_name]["ft_zero_point"]
                for input_name in exc_info["input"]:
                    self.input_scale=self.quant_observer_table[input_name]["ft_scale"]
                    self.input_zero_point=self.quant_observer_table[input_name]["ft_zero_point"]
                    self.rescale=self.output_scale/self.input_scale
                    if self.rescale==1:
                        exec("self.add_list.append({var_name})".format(var_name=input_name))
                    else:
                        exec("self.add_list.append(({var_name}-self.input_zero_point)/self.rescale)".format(var_name=input_name))
                
                exec("{var_name}=torch.add(self.add_list[0],self.add_list[1])".format(var_name=output_name))
                exec("{var_name}={var_name}+self.output_zero_point".format(var_name=output_name))
                exec("{var_name}={var_name}.long().float()".format(var_name=output_name))
                exec("torch.clamp({var_name},min=self.QUANT_MIN,max=self.QUANT_MAX)".format(var_name=output_name))
            # elif op_type!="Identity":
            #     # output_name=exc_info["output"][0]
            #     exec(exc_info["exec"])
                # exec("{var_name}={var_name}.long().float()".format(var_name=output_name))
            else:
                exec(exc_info["exec"])

        if len(net_output)==1:
            exc_str="self.net_output={node_output}".format(node_output=self.onnxBlobNameTable[net_output[0].name])
            exec(exc_str)
        return self.net_output


    def quantilize(self,convert_weights=False):
        self.convert_weights=convert_weights
        if self.quant_method=="kl":
            self.calculateKL()
        # calculate scale
        quant_observer=self.input_observer["quant_observer"]

        ft_abs_float_max=max(abs(quant_observer["ft_float_min"]),abs(quant_observer["ft_float_max"]))
        ft_scale,ft_zero_point=self.calculateScaleZeroPoint(float_min=quant_observer["ft_float_min"],\
            float_max=quant_observer["ft_float_max"],quant_mode=self.ft_quant_mode,quant_granularity="total")
        
        self.input_observer["quant_observer"]["ft_scale"]=ft_scale
        self.input_observer["quant_observer"]["ft_zero_point"]=ft_zero_point
        self.input_observer["id"]=-1

        for exc_info in self.forwardExcList:
            op_type=exc_info["type"]
            if op_type in self.QUANT_OP_LIST:
                if op_type=="Conv":
                    output_name=exc_info["output"][0]
                    next_op_type=self.getNextTypeByInput(output_name)
                    if(self.checkActiInPatch(next_op_type)):
                        act_exc_info=self.getInfoByType(next_op_type)
                        quant_observer=act_exc_info["quant_observer"]
                        ft_abs_float_max=max(abs(quant_observer["ft_float_min"]),abs(quant_observer["ft_float_max"]))
                        ft_scale,ft_zero_point=self.calculateScaleZeroPoint(float_min=quant_observer["ft_float_min"],\
                            float_max=quant_observer["ft_float_max"],quant_mode=self.ft_quant_mode,quant_granularity="total")
                        exc_info["quant_observer"]["ft_scale"]=ft_scale
                        exc_info["quant_observer"]["ft_zero_point"]=ft_zero_point
                    else:
                        quant_observer=exc_info["quant_observer"]
                        ft_abs_float_max=max(abs(quant_observer["ft_float_min"]),abs(quant_observer["ft_float_max"]))
                        ft_scale,ft_zero_point=self.calculateScaleZeroPoint(float_min=quant_observer["ft_float_min"],\
                            float_max=quant_observer["ft_float_max"],quant_mode=self.ft_quant_mode,quant_granularity="total")
                        exc_info["quant_observer"]["ft_scale"]=ft_scale
                        exc_info["quant_observer"]["ft_zero_point"]=ft_zero_point
                else:
                    quant_observer=exc_info["quant_observer"]
                    ft_abs_float_max=max(abs(quant_observer["ft_float_min"]),abs(quant_observer["ft_float_max"]))
                    ft_scale,ft_zero_point=self.calculateScaleZeroPoint(float_min=quant_observer["ft_float_min"],\
                        float_max=quant_observer["ft_float_max"],quant_mode=self.ft_quant_mode,quant_granularity="total")
                    exc_info["quant_observer"]["ft_scale"]=ft_scale
                    exc_info["quant_observer"]["ft_zero_point"]=ft_zero_point

        for exc_info in self.forwardExcList:
            op_type=exc_info["type"]
            if op_type =="Conv":
                var_name=exc_info["var_name"]

                if self.quant_granularity=="channel":
                    exec("self.weight_clone={var_name}.weight.data.clone()".format(var_name=var_name))
                    self.weight_clone=self.weight_clone.view(self.weight_clone.shape[0],-1)
                    self.wt_float_min,indice=torch.min(self.weight_clone,dim=1)
                    self.wt_float_max,indice=torch.max(self.weight_clone,dim=1)
                    self.wt_float_min=self.wt_float_min.cpu().numpy()
                    self.wt_float_max=self.wt_float_max.cpu().numpy()
                else:
                    exec("self.wt_float_min=torch.min({var_name}.weight.data).cpu().numpy()".format(var_name=var_name))
                    exec("self.wt_float_max=torch.max({var_name}.weight.data).cpu().numpy()".format(var_name=var_name))
                wt_scale,wt_zero_point=self.calculateScaleZeroPoint(float_min=self.wt_float_min,\
                    float_max=self.wt_float_min,quant_mode=self.wt_quant_mode,quant_granularity=self.quant_granularity)
                exc_info["quant_observer"]["wt_scale"]=wt_scale
                exc_info["quant_observer"]["wt_zero_point"]=wt_zero_point

            elif op_type =="Constant":
                var_name=exc_info["var_name"]
                exec("self.wt_float_min=torch.min({var_name}.data).cpu().numpy()".format(var_name=var_name))
                exec("self.wt_float_max=torch.max({var_name}.data).cpu().numpy()".format(var_name=var_name))
                wt_scale,wt_zero_point=self.calculateScaleZeroPoint(float_min=self.wt_float_min,\
                    float_max=self.wt_float_min,quant_mode=self.wt_quant_mode,quant_granularity="total")
                exc_info["quant_observer"]["wt_scale"]=wt_scale
                exc_info["quant_observer"]["wt_zero_point"]=wt_zero_point

        self.summerizeQuantInfo()
        # print(self.quant_observer_table)
        if self.convert_weights:
            self.convertWeightsFix()

    def convertWeightsFix(self):
        # print(self.quant_observer_table)
        for exc_info in self.forwardExcList:
            op_type=exc_info["type"]
            if op_type =="Conv":
                var_name=exc_info["var_name"]
                input_name=exc_info["input"][0]
                output_name=exc_info["output"][0]
                self.wt_scale=exc_info["quant_observer"]["wt_scale"]
                self.wt_zero_point=exc_info["quant_observer"]["wt_zero_point"]
                self.ft_scale=exc_info["quant_observer"]["ft_scale"]
                self.ft_zero_point=exc_info["quant_observer"]["ft_zero_point"]
                self.input_ft_scale=self.quant_observer_table[input_name]["ft_scale"]
                self.input_ft_zero_point=self.quant_observer_table[input_name]["ft_zero_point"]
                self.scale_mul=self.wt_scale*self.input_ft_scale

                if self.quant_granularity=="channel":
                    exec("self.weight_clone={var_name}.weight.data.clone()".format(var_name=var_name))
                    for i in range(self.weight_clone.shape[0]):
                        self.weight_clone[i]/=self.wt_scale[i]+self.wt_zero_point[i]
                    exec("{var_name}.weight.data=self.weight_clone".format(var_name=var_name))
                else:
                    exec("{var_name}.weight.data/=self.wt_scale+self.wt_zero_point".format(var_name=var_name))
                
                exec("{var_name}.weight.data={var_name}.weight.data.long().float()".format(var_name=var_name))
                
                # ft zero point bias trans
                exec("self.weight_clone={var_name}.weight.data.clone()".format(var_name=var_name))
                self.weight_sumup=torch.sum(self.weight_clone,dim=(1,2,3))
                self.quant_bias=-self.weight_sumup*self.input_ft_zero_point
                self.quant_bias=self.quant_bias.long().float().view(1,-1,1,1)
                exec("{output_name}_quant_bias=self.quant_bias".format(output_name=output_name))
                quantExcStr="{output_name}+={output_name}_quant_bias".format(output_name=output_name)
                exc_info["quant_add_bias_exc"]=quantExcStr

                if exc_info["has_bias"]:
                    exec("{var_name}.bias.data/=self.scale_mul".format(var_name=var_name))
                    exec("{var_name}.bias.data={var_name}.bias.data.long().float()".format(var_name=var_name))
                
                # redefine padding
                node=get_node_by_input(self.getOnnxNameFromTable(input_name),self.nodes)
                attr=attribute_to_dict(node.attribute)
                pad_t=attr["pads"][0]
                pad_b=attr["pads"][2]
                pad_l=attr["pads"][1]
                pad_r=attr["pads"][3]
                exc_str_pad="{var_name}_pad=nn.ConstantPad2d(padding={padding},value={value})".format(var_name=var_name,padding=(pad_l,pad_r,pad_t,pad_b),value=self.input_ft_zero_point)
                exec(exc_str_pad)
                # print(attr)

            elif op_type =="MaxPool":
                var_name=exc_info["var_name"]
                input_name=exc_info["input"][0]
                output_name=exc_info["output"][0]
                self.input_ft_zero_point=self.quant_observer_table[input_name]["ft_zero_point"]
                # redefine padding
                node=get_node_by_input(self.getOnnxNameFromTable(input_name),self.nodes)
                attr=attribute_to_dict(node.attribute)
                pad_t=attr["pads"][0]
                pad_b=attr["pads"][2]
                pad_l=attr["pads"][1]
                pad_r=attr["pads"][3]
                exc_str_pad="{var_name}_pad=nn.ConstantPad2d(padding={padding},value={value})".format(var_name=var_name,padding=(pad_l,pad_r,pad_t,pad_b),value=self.input_ft_zero_point)
                exec(exc_str_pad)

            elif op_type =="MatMul":
                var_name=exc_info["var_name"]
                input_name_0=exc_info["input"][0]
                input_name_1=exc_info["input"][1]
                output_name=exc_info["output"][0]
                self.input_ft_scale=self.quant_observer_table[input_name_0]["ft_scale"]
                self.input_ft_zero_point=self.quant_observer_table[input_name_0]["ft_zero_point"]
                self.wt_scale=self.quant_observer_table[input_name_1]["wt_scale"]
                self.wt_zero_point=self.quant_observer_table[input_name_1]["wt_zero_point"]
                self.output_ft_scale=self.quant_observer_table[output_name]["ft_scale"]
                self.output_ft_zero_point=self.quant_observer_table[output_name]["ft_zero_point"]
                # ft zero point bias trans
                exec("self.weight_clone={var_name}.clone()".format(var_name=input_name_1))
                self.weight_sumup=torch.sum(self.weight_clone,dim=(0))
                self.quant_bias=-self.weight_sumup*self.input_ft_zero_point
                self.quant_bias=self.quant_bias.long().float().view(1,-1)
                # print("quant bias shape:",self.quant_bias.shape)
                exec("{output_name}_quant_bias=self.quant_bias".format(output_name=output_name))
                quantExcStr="{output_name}+={output_name}_quant_bias".format(output_name=output_name)
                exc_info["quant_add_bias_exc"]=quantExcStr

            elif op_type =="Constant":
                var_name=exc_info["var_name"]
                self.wt_scale=exc_info["quant_observer"]["wt_scale"]
                self.wt_zero_point=exc_info["quant_observer"]["wt_zero_point"]
                exec("{var_name}.data/=self.wt_scale+self.wt_zero_point".format(var_name=var_name))
                exec("{var_name}.data={var_name}.data.long().float()".format(var_name=var_name))

            elif op_type =="Relu":
                var_name=exc_info["var_name"]
                input_name=exc_info["input"][0]
                output_name=exc_info["output"][0]
                self.input_ft_scale=self.quant_observer_table[input_name]["ft_scale"]
                self.input_ft_zero_point=self.quant_observer_table[input_name]["ft_zero_point"]
                exec("{input_name}_zero_point=self.input_ft_zero_point".format(input_name=input_name))
                quantExcStr="{output_name}=self.q_relu({input_name},{input_name}_zero_point)".format(output_name=output_name,input_name=input_name)
                exc_info["exec"]=quantExcStr
                
    def summerizeQuantInfo(self):
        output_name=self.input_observer["output"][0]
        self.quant_observer_table[output_name]=self.input_observer["quant_observer"]

        for exc_info in self.forwardExcList:
            op_type=exc_info["type"]
            
            if op_type=="Conv":
                assert(len(exc_info["output"])==1)
                output_name=exc_info["output"][0]
                self.quant_observer_table[output_name]=exc_info["quant_observer"]

            elif op_type=="MatMul":
                assert(len(exc_info["output"])==1)
                output_name=exc_info["output"][0]
                self.quant_observer_table[output_name]=exc_info["quant_observer"]

            elif op_type=="Add":
                assert(len(exc_info["output"])==1)
                assert(len(exc_info["input"])==2)
                output_name=exc_info["output"][0]
                input_name=exc_info["input"]
                self.quant_observer_table[output_name]=exc_info["quant_observer"]
                # scale_0=self.quant_observer_table[input_name[0]]["ft_scale"]
                # scale_1=self.quant_observer_table[input_name[1]]["ft_scale"]
                
                # if scale_0<scale_1:
                #     self.quant_observer_table[output_name]=self.quant_observer_table[input_name[0]]
                # else:
                #     self.quant_observer_table[output_name]=self.quant_observer_table[input_name[1]]

            elif op_type=="Constant":
                assert(len(exc_info["output"])==1)
                output_name=exc_info["output"][0]
                wt_scale=exc_info["quant_observer"]["wt_scale"]
                wt_zero_point=exc_info["quant_observer"]["wt_zero_point"]
                exc_info["quant_observer"]["ft_scale"]=wt_scale
                exc_info["quant_observer"]["ft_zero_point"]=wt_zero_point
                self.quant_observer_table[output_name]=exc_info["quant_observer"]

            else:
                assert(len(exc_info["output"])==1)
                assert(len(exc_info["input"])==1)
                output_name=exc_info["output"][0]
                input_name=exc_info["input"][0]
                self.quant_observer_table[output_name]=self.quant_observer_table[input_name]
    
    def findExcInfoByOutput(self,input_name):
        if input_name in self.input_observer["output"]:
            return self.input_observer
        for exc_info in self.forwardExcList:
            if input_name in exc_info["output"]:
                return exc_info

    def mse(self,v1,v2):
        assert(v1.size==v2.size)
        return np.sum((v1-v2)*(v1-v2))/v2.size

    def dumpMse(self):
        for exc_info in self.forwardExcList:
            op_type=exc_info["type"]

            if(op_type =="Softmax" or op_type =="Identity"):
                var_name=exc_info["var_name"]
                output_name=exc_info["output"][0]
                quant_result=self.getFeatureTensor(output_name).cpu().numpy()
                float_result=self.float_model.getFeatureTensor(output_name).cpu().numpy()
                print(var_name,output_name," mse:",self.mse(quant_result,float_result))

            elif op_type !="Constant":
                var_name=exc_info["var_name"]
                output_name=exc_info["output"][0]
                self.ft_scale=self.quant_observer_table[output_name]["ft_scale"]
                self.ft_zero_point=self.quant_observer_table[output_name]["ft_zero_point"]

                quant_result=(self.getFeatureTensor(output_name).cpu().numpy()-self.ft_zero_point)*self.ft_scale
                float_result=self.float_model.getFeatureTensor(output_name).cpu().numpy()
                print(var_name,output_name,self.ft_scale,self.ft_zero_point," mse:",self.mse(quant_result,float_result))

    def checkActiInPatch(self,act_type):
        for patch in self.quant_patch:
            if act_type in patch:
                return True
        return False

    def getInfoByType(self,op_type):
        for exc_info in self.forwardExcList:
            if op_type==exc_info["type"]:
                return exc_info


    def getNextTypeByInput(self,input_blob_name):
        for exc_info in self.forwardExcList:
            input_blob=exc_info["input"]
            op_type=exc_info["type"]
            if input_blob_name in input_blob:
                return op_type