# from __future__ import absolute_import
import torch
import torch.nn as nn
import onnx
from typing import List, Dict, Union, Optional, Tuple, Sequence
import copy
from .util import*
from torch.autograd import Variable

class onnxTorchModel(nn.Module):
    def __init__(self,onnx_model: onnx.ModelProto,cfg:dict):
        super(onnxTorchModel,self).__init__()
        self.onnx_model=onnx_model
        self.nodes=self.onnx_model.graph.node
        self.pad_split=cfg["pad_split"]

        self.weights_in_constant_flg=False
        if len(onnx_model.graph.initializer)==0:
            self.weights_in_constant_flg=True

        self.op_type_list=[]
        self.current_id=0
        self.forwardExcList=[]
        self.onnxBlobNameTable={}

        self.generateOnnxBlobNameTable()
        self.parseOnnx()
        
    def getOnnxNameFromTable(self,name):
        for n in self.onnxBlobNameTable.keys():
            if self.onnxBlobNameTable[n]==name:
                return n

    def forward(self, input):
        net_input=self.onnx_model.graph.input
        net_output=self.onnx_model.graph.output
        if len(net_input)==1:
            exc_str="{node_input}=input".format(node_input=self.onnxBlobNameTable[net_input[0].name])
            exec(exc_str)
            
        for exc_info in self.forwardExcList:
            if "exec_pad" in exc_info.keys():
                exec(exc_info["exec_pad"])
            exc_str=exc_info["exec"]
            exec(exc_str)
            
        if len(net_output)==1:
            exc_str="self.net_output={node_output}".format(node_output=self.onnxBlobNameTable[net_output[0].name])
            exec(exc_str)
        return self.net_output

    def parseOnnx(self):
        nodes = self.onnx_model.graph.node
        for nid,node in enumerate(nodes):
            self.current_id=nid
            op_type=node.op_type
            if op_type not in self.op_type_list:
                self.op_type_list.append(op_type)
            print("Parsing onnx:",op_type)

            if op_type=="Conv":
                self.parseConv(node)
            elif op_type=="BatchNormalization":
                self.parseBN(node)
            elif op_type=="Flatten":
                self.parseFlatten(node)
            elif op_type=="Relu":
                self.parseRelu(node)
            elif op_type=="MaxPool":
                self.parseMaxPool(node)
            elif op_type=="Add":
                self.parseAdd(node)
            elif op_type=="GlobalAveragePool":
                self.parseGlobalAveragePool(node)
            elif op_type=="MatMul":
                self.parseMatMul(node)
            elif op_type=="Softmax":
                self.parseSoftmax(node)
            elif op_type=="Identity":
                self.parseIdentity(node)
            elif op_type=="Constant":
                self.parseNonWeightsConstant(node)


    # torch.nn.Conv2d(in_channels: int, out_channels: int, 
    # kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, 
    # padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, 
    # groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')
    def parseConv(self,node):
        attr=attribute_to_dict(node.attribute)
        if(self.weights_in_constant_flg):
            wt,bt=get_conv_params_in_constant(node,self.onnx_model.graph.node)
        has_bias=True
        if len(node.input)==2:
            has_bias=False
        c,n,k_w,k_h=wt.shape
        c=c*int(attr["group"])
        n=n*int(attr["group"])
        var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
        pad_t=attr["pads"][0]
        pad_b=attr["pads"][2]
        pad_l=attr["pads"][1]
        pad_r=attr["pads"][3]
        
        if(pad_t!=pad_b or pad_l!=pad_r or self.pad_split):
            exc_str_pad="{var_name}_pad=nn.ConstantPad2d(padding={padding},value={value})".format(var_name=var_name,padding=(pad_l,pad_r,pad_t,pad_b),value=0)
            exc_str_conv="{var_name}=nn.Conv2d(in_channels={in_channels},out_channels={out_channels},kernel_size={kernel_size},stride={stride},padding={padding},dilation={dilation},groups={groups},bias={bias})".format(var_name=var_name,\
                in_channels=c,\
                out_channels=n,\
                kernel_size=tuple(attr["kernel_shape"]),\
                stride=tuple(attr["strides"]),\
                padding=(0,0),\
                dilation=tuple(attr["dilations"]),\
                groups=attr["group"],\
                bias=True)
            
            self.generateForwardExec(node,var_name,op_pad_split=True)
            exec(exc_str_pad)
            exec(exc_str_conv)
            exc_init_weights_str="{var_name}.weight=torch.nn.Parameter(torch.Tensor(wt))".format(var_name=var_name)
            exec(exc_init_weights_str)
        else:
            exc_str="{var_name}=nn.Conv2d(in_channels={in_channels},out_channels={out_channels},kernel_size={kernel_size},stride={stride},padding={padding},dilation={dilation},groups={groups},bias={bias})".format(var_name=var_name,\
                            in_channels=c,\
                            out_channels=n,\
                            kernel_size=tuple(attr["kernel_shape"]),\
                            stride=tuple(attr["strides"]),\
                            padding=tuple(attr["pads"][:2]),\
                            dilation=tuple(attr["dilations"]),\
                            groups=attr["group"],\
                            bias=True)

            self.generateForwardExec(node,var_name)
            exec(exc_str)
            exc_init_weights_str="{var_name}.weight=torch.nn.Parameter(torch.Tensor(wt))".format(var_name=var_name)
            exec(exc_init_weights_str)


        if has_bias:
            self.forwardExcList[len(self.forwardExcList)-1]["has_bias"]=True
            exc_init_bias_str="{var_name}.bias=torch.nn.Parameter(torch.Tensor(bt))".format(var_name=var_name)
            exec(exc_init_bias_str)
        else:
            self.forwardExcList[len(self.forwardExcList)-1]["has_bias"]=False
            exc_init_bias_str="nn.init.constant_({var_name}.bias, 0)".format(var_name=var_name)
            exec(exc_init_bias_str)

    # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def parseBN(self,node):
        attr=attribute_to_dict(node.attribute)
        if(self.weights_in_constant_flg):
            bn_scale,bn_B,bn_mean,bn_var=get_bn_params_in_constant(node,self.onnx_model.graph.node)
        n=bn_scale.shape[0]
        var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
        exc_str="{var_name}=nn.BatchNorm2d(num_features={num_features},eps={eps},momentum={momentum})".format(var_name=var_name,\
            num_features=n,eps=attr["epsilon"],momentum=attr["momentum"])
        exec(exc_str)
        bn_scale,bn_B,bn_mean,bn_var=get_bn_params_in_constant(node, self.nodes)
        exc_init_scale_str="{var_name}.weight=torch.nn.Parameter(torch.Tensor(bn_scale))".format(var_name=var_name)
        exc_init_bias_str="{var_name}.bias=torch.nn.Parameter(torch.Tensor(bn_B))".format(var_name=var_name)
        exc_init_mean_str="{var_name}.running_mean=torch.Tensor(bn_mean)".format(var_name=var_name)
        exc_init_var_str="{var_name}.running_var=torch.Tensor(bn_var)".format(var_name=var_name)
        exec(exc_init_scale_str)
        exec(exc_init_bias_str)
        exec(exc_init_mean_str)
        exec(exc_init_var_str)
        self.generateForwardExec(node,var_name)

    def parseFlatten(self,node):
        attr=attribute_to_dict(node.attribute)
        var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
        exc_str="{var_name}=nn.Flatten(start_dim={start_dim})".format(var_name=var_name,start_dim=attr["axis"])
        self.generateForwardExec(node,var_name)
        exec(exc_str)

    def parseRelu(self,node):
        attr=attribute_to_dict(node.attribute)
        var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
        exc_str="{var_name}=nn.ReLU()".format(var_name=var_name)
        self.generateForwardExec(node,var_name)
        exec(exc_str)

    # torch.nn.MaxPool2d(kernel_size: Union[T, Tuple[T, ...]], 
    # stride: Optional[Union[T, Tuple[T, ...]]] = None, 
    # padding: Union[T, Tuple[T, ...]] = 0, dilation: Union[T, Tuple[T, ...]] = 1, 
    # return_indices: bool = False, ceil_mode: bool = False)
    def parseMaxPool(self,node):
        attr=attribute_to_dict(node.attribute)
        var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
        pad_t=attr["pads"][0]
        pad_b=attr["pads"][2]
        pad_l=attr["pads"][1]
        pad_r=attr["pads"][3]
        
        if(pad_t!=pad_b or pad_l!=pad_r or pad_r!=pad_t or self.pad_split):
            exc_str_pad="{var_name}_pad=nn.ConstantPad2d(padding={padding},value={value})".format(var_name=var_name,padding=(pad_l,pad_r,pad_t,pad_b),value=0)
            exc_str="{var_name}=nn.MaxPool2d(kernel_size={kernel_shape},padding={pads},stride={strides})".format(var_name=var_name,\
                kernel_shape=tuple(attr["kernel_shape"]),\
                pads=0,\
                strides=tuple(attr["strides"]))
            exec(exc_str_pad)
            exec(exc_str)
            self.generateForwardExec(node,var_name,op_pad_split=True)
        else:
            exc_str="{var_name}=nn.MaxPool2d(kernel_size={kernel_shape},padding={pads},stride={strides})".format(var_name=var_name,\
                kernel_shape=tuple(attr["kernel_shape"]),\
                pads=attr["pads"][0],\
                strides=tuple(attr["strides"]))
            exec(exc_str)
            self.generateForwardExec(node,var_name)
        

    def parseAdd(self,node):
        attr=attribute_to_dict(node.attribute)
        var_name="torch.add"
        self.generateForwardExecMultiInput(node,var_name,filter_const=False,is_instance=False)

    def parseGlobalAveragePool(self,node):
        attr=attribute_to_dict(node.attribute)
        var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
        exc_str="{var_name}=nn.AdaptiveAvgPool2d((1, 1))".format(var_name=var_name)
        self.generateForwardExec(node,var_name)
        exec(exc_str)

    def parseMatMul(self,node):
        attr=attribute_to_dict(node.attribute)
        var_name="torch.matmul"
        self.generateForwardExecMultiInput(node,var_name,filter_const=False,is_instance=False)

    def parseSoftmax(self,node):
        attr=attribute_to_dict(node.attribute)
        var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
        if attr["axis"]==-1:
            exc_str="{var_name}=nn.Softmax(dim=1)".format(var_name=var_name)
            exec(exc_str)
        else:
            exc_str="{var_name}=nn.Softmax(dim={dim})".format(var_name=var_name,dim= attr["axis"])
            exec(exc_str)
        self.generateForwardExec(node,var_name)
        
    def parseIdentity(self,node):
        inputs=node.input
        outputs=node.output
        var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
        input_blob=self.onnxBlobNameTable[inputs[0]]
        output_blob=self.onnxBlobNameTable[outputs[0]]
        forwardExcStr="{output_name}={input_name}".format(output_name=output_blob,input_name=input_blob)
        nodeInfoDict={"exec":forwardExcStr,"var_name":var_name,"type":"Identity","input":[input_blob],"output":[output_blob],"is_instance":False,"id":self.current_id}
        self.forwardExcList.append(nodeInfoDict)

    def parseNonWeightsConstant(self,node):
        output_name=node.output[0]
        next_type=get_node_type_by_input(output_name,self.nodes)
        weight_node_list=["Conv","BatchNormalization"]
        if next_type not in weight_node_list:
            constant_tonser=get_tensor_in_constant(output_name,self.nodes)
            var_name="self.{type}_{id}".format(type=node.op_type,id=self.current_id)
            output_blob=self.onnxBlobNameTable[output_name]
            exc_str="{var_name}=torch.nn.Parameter(torch.tensor(constant_tonser))".format(var_name=var_name)
            exec(exc_str)
            forwardExcStr="{output}={var_name}".format(output=output_blob,var_name=var_name)
            nodeInfoDict={"exec":forwardExcStr,"var_name":var_name,"type":node.op_type,"input":[],"output":[output_blob],"is_instance":True}
            self.forwardExcList.append(nodeInfoDict)

    ###################################### support func area
    def generateForwardExec(self,node,var_name,filter_const=True,is_instance=True,op_pad_split=False):
        inputs=node.input
        outputs=node.output
        # node_type=node.op_type
        # next_type=
        dynamic_input=[]
        dynamic_output=[]

        for inputname in inputs:
            if filter_const and get_node_type_by_output(inputname,self.nodes)=="Constant":
                continue
            dynamic_input.append(self.onnxBlobNameTable[inputname])

        for outputname in outputs:
            dynamic_output.append(self.onnxBlobNameTable[outputname])

        if len(dynamic_input)>1:
            assert(0)

        if len(dynamic_input)==0:
            dynamic_input.append(self.onnxBlobNameTable[inputs[0]])

        input_blob=dynamic_input[0]
        output_blob=dynamic_output[0]

        if op_pad_split:
            forwardExcStrPad="{output_name}_pad={var_name}_pad({input_name})".format(output_name=input_blob,var_name=var_name,input_name=input_blob)
            forwardExcStr="{output_name}={var_name}({input_name}_pad)".format(output_name=output_blob,var_name=var_name,input_name=input_blob)
            nodeInfoDict={"exec":forwardExcStr,"exec_pad":forwardExcStrPad,"var_name":var_name,"type":node.op_type,"input":dynamic_input,"output":[output_blob],"is_instance":is_instance,"id":self.current_id}
        else:
            forwardExcStr="{output_name}={var_name}({input_name})".format(output_name=output_blob,var_name=var_name,input_name=input_blob)
            nodeInfoDict={"exec":forwardExcStr,"var_name":var_name,"type":node.op_type,"input":dynamic_input,"output":[output_blob],"is_instance":is_instance,"id":self.current_id}
        
        self.forwardExcList.append(nodeInfoDict)

        for i in range(1,len(dynamic_output)):
            forwardExcStr="{output_name}={input_name}".format(output_name=dynamic_output[i],input_name=dynamic_output[0])
            nodeInfoDict={"exec":forwardExcStr,"var_name":"Copy","type":"Copy","input":[dynamic_output[0]],"output":[output_blob],"is_instance":False,"id":self.current_id}
            self.forwardExcList.append(nodeInfoDict)

    def generateForwardExecMultiInput(self,node,var_name,filter_const=True,is_instance=True):
        inputs=node.input
        outputs=node.output
        dynamic_input=[]
        dynamic_output=[]

        for inputname in inputs:
            if filter_const and get_node_type_by_output(inputname,self.nodes)=="Constant":
                continue
            dynamic_input.append(self.onnxBlobNameTable[inputname])

        for outputname in outputs:
            dynamic_output.append(self.onnxBlobNameTable[outputname])

        input_blob=dynamic_input[0]
        output_blob=dynamic_output[0]

        input_blob_str=""
        for input_blob in dynamic_input:
            input_blob_str+=","+input_blob

        input_blob_str=input_blob_str[1:]
        forwardExcStr="{output_name}={var_name}({input_name})".format(output_name=output_blob,var_name=var_name,input_name=input_blob_str)
        nodeInfoDict={"exec":forwardExcStr,"var_name":var_name,"type":node.op_type,"input":dynamic_input,"output":[output_blob],"is_instance":is_instance,"id":self.current_id}
        self.forwardExcList.append(nodeInfoDict)

        for i in range(1,len(dynamic_output)):
            forwardExcStr="{output_name}={input_name}".format(output_name=dynamic_output[i],input_name=dynamic_output[0])
            nodeInfoDict={"exec":forwardExcStr,"var_name":"Copy","type":"Copy","input":[dynamic_output[0]],"output":[output_blob],"is_instance":False,"id":self.current_id}
            self.forwardExcList.append(nodeInfoDict)

    def generateOnnxBlobNameTable(self):
        nodes = self.onnx_model.graph.node
        id_count=0
        for nid,node in enumerate(nodes):
            inputs=node.input
            outputs=node.output
            for name in inputs:
                if name not in self.onnxBlobNameTable.keys():
                    self.onnxBlobNameTable[name]="self.blob_"+str(id_count)
                    id_count+=1
            for name in outputs:
                if name not in self.onnxBlobNameTable.keys():
                    self.onnxBlobNameTable[name]="self.blob_"+str(id_count)
                    id_count+=1
    
    def getFeatureTensor(self,name):
        exec("self.outTensor= {name}".format(name=name))
        return self.outTensor