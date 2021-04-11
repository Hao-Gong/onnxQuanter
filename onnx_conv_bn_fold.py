import onnx
from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto, version_converter
import sys
import logging
from onnx import optimizer
from onnx import numpy_helper
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[ONNX_AR_OPTIMIZER]")
ESP=0.0000000001

ONNX_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}

def infer_model_shape(onnx_model):
    tensors = onnx_model.graph.initializer
    for i, tensor in enumerate(tensors):
        value_info = helper.make_tensor_value_info(tensor.name, ONNX_DTYPE[tensor.data_type], tensor.dims)
        onnx_model.graph.input.insert(i+1, value_info) # because 0 is for placeholder, so start index is 1
    inferred_onnx_model = shape_inference.infer_shapes(onnx_model)

    return inferred_onnx_model

def save_model(onnx_model, path):
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, path)
    
def get_node_by_output_name(name, nodes):
    return_nodes = []
    for node in nodes:
        if name in node.output:
            return_nodes.append(node)
    return return_nodes
from onnx import numpy_helper
def get_node_by_input_name(names, nodes):
    return_nodes = []
    for node in nodes:
        for name in names:
            if name in node.input:
                return_nodes.append(node)
    return return_nodes

def get_node_by_output_name(names, nodes):
    return_nodes = []
    for node in nodes:
        for name in names:
            if name in node.output:
                return_nodes.append(node)
    return return_nodes

def get_value_info_by_name(name, model):
    for x in model.graph.value_info:
        if x.name == name:
            return x
    return None

def get_tensor_in_constant(name,nodes):
    for node in nodes:
        if node.op_type=="Constant":
            if node.output==[name]:
                return numpy_helper.to_array(node.attribute[0].t)
    return None

def get_bn_params_in_constant(node,nodes):
    if not node.op_type=="BatchNormalization":
        print("Not BatchNormalization")
        return "Not BatchNormalization"
    
    bn_scale=get_tensor_in_constant(node.input[1],nodes)
    bn_B=get_tensor_in_constant(node.input[2],nodes)
    bn_mean=get_tensor_in_constant(node.input[3],nodes)
    bn_var=get_tensor_in_constant(node.input[4],nodes)
    return bn_scale,bn_B,bn_mean,bn_var

def get_conv_params_in_constant(node,nodes):
    if not node.op_type=="Conv":
        print("Not Conv")
        return "Not Conv"
    
    conv_W=get_tensor_in_constant(node.input[1],nodes)
    if len(node.input)==2:
        conv_B=None
    else:
        conv_B=get_tensor_in_constant(node.input[2],nodes)
    return conv_W,conv_B

def attribute_to_dict(attribute):
    attri_dict={}
    for att in attribute:
        attri_dict[att.name]=helper.get_attribute_value(att)
    return attri_dict

def create_onnx_tensor_info(shape, dtype, tensor_name):
    return make_tensor_value_info(
        tensor_name, dtype, create_onnx_shape(shape))

def conv_bn_fold(model):
    nodes = model.graph.node
    node_delete_queque=[]
    node_conv_replace_queque=[]
    node_add_queque=[]
    for node_id,node in enumerate(nodes):
        if node.op_type=="Conv":
            output_name=node.output
            next_nodes=get_node_by_input_name(output_name, nodes)
            if next_nodes[0].op_type=="BatchNormalization":
                logger.info("Folding "+node.name+" "+next_nodes[0].name)
                attribute=node.attribute
                recrusive_stop_flg=0
                bn_node=next_nodes[0]
                bn_scale,bn_B,bn_mean,bn_var=get_bn_params_in_constant(bn_node, nodes)
                conv_W,conv_B=get_conv_params_in_constant(node,nodes)
                bn_scale_reshape=bn_scale.reshape((bn_scale.shape[0],1,1,1))
                bn_var_reshape=bn_var.reshape((bn_var.shape[0],1,1,1))
                conv_W_fold=bn_scale_reshape*conv_W/np.sqrt(bn_var_reshape+ESP)
                
                if conv_B==None:
                    conv_B_fold=bn_B-bn_scale*bn_mean/np.sqrt(bn_var+ESP)
                # delete bn and bn params
                node_delete_queque.append(bn_node)
                # node_delete_queque.append(node)
                for input_name in bn_node.input:
                    for n in nodes:
                        if n.op_type=="Constant" and n.output==[input_name]:
                            node_delete_queque.append(n)

                # delete conv constants params
                for input_name in node.input:
                    for n in nodes:
                        if n.op_type=="Constant" and n.output==[input_name]:
                            node_delete_queque.append(n)

                conv_bn_fold_b_name=bn_node.input[2]
                node_input=node.input

                conv_bn_fold_input=[node_input[0],node_input[1],conv_bn_fold_b_name]
                conv_bn_fold_weight_constant_node=onnx.helper.make_node(\
                    op_type="Constant",\
                    inputs=[],\
                    outputs=[conv_bn_fold_input[1]],\
                    value=onnx.helper.make_tensor(conv_bn_fold_input[1], onnx.TensorProto.FLOAT, [conv_W.shape[0],conv_W.shape[1],conv_W.shape[2],conv_W.shape[3]],\
                         conv_W_fold.reshape(-1)))

                node_add_queque.append({"node":conv_bn_fold_weight_constant_node})
                conv_bn_fold_bias_constant_node=onnx.helper.make_node(\
                    op_type="Constant",\
                    inputs=[],\
                    outputs=[conv_bn_fold_input[2]],\
                    value=onnx.helper.make_tensor(conv_bn_fold_input[2], onnx.TensorProto.FLOAT, [conv_B_fold.shape[0]],\
                         conv_B_fold.reshape(-1)))
                node_add_queque.append({"node":conv_bn_fold_bias_constant_node})

                attribute_dict=attribute_to_dict(attribute)

                conv_bn_fold_node=helper.make_node(
                    op_type="Conv",\
                    name=node.name,\
                    inputs=conv_bn_fold_input,\
                    outputs=bn_node.output,\
                    **attribute_dict)
                node_conv_replace_queque.append({"node_id":node_id,"ori_conv":node,"fold_conv":conv_bn_fold_node})

        elif node.op_type=="Flatten":
            input_name=node.input
            output_name=node.output
            # print(node)
            output_nodes=get_node_by_input_name(output_name, nodes)
            input_nodes=get_node_by_output_name(input_name, nodes)
            # print(input_nodes[0].op_type,output_nodes[0].op_type)
            if input_nodes[0].op_type=="Constant" and output_nodes[0].op_type=="MatMul":
                node_delete_queque.append(node)
                new_matmul_input=[output_nodes[0].input[0],input_nodes[0].output[0]]
                # print()
                new_matmul_node=helper.make_node(
                    op_type="MatMul",\
                    name=output_nodes[0].name,\
                    inputs=new_matmul_input,\
                    outputs=output_nodes[0].output)
                node_conv_replace_queque.append({"node_id":node_id,"ori_conv":output_nodes[0],"fold_conv":new_matmul_node})

    # print(node_add_queque)
    for ele in node_conv_replace_queque:
        node_id=ele["node_id"]
        del_conv_node=ele["ori_conv"]
        add_conv_node=ele["fold_conv"]
        nodes.remove(del_conv_node)
        nodes.insert(node_id,add_conv_node)

    for del_node in node_delete_queque:
        nodes.remove(del_node)

    for ele in node_add_queque:
        add_node=ele["node"]
        if add_node.op_type=="Constant":
            nodes.insert(0,add_node)
    return model

def replace_strange_symble(model, strange_symbol_reorientation_table):
    # for node_name in end_node_names:
    nodes = model.graph.node
    # print(nodes)
    for node in nodes:
        for src in strange_symbol_reorientation_table.keys():
            target=strange_symbol_reorientation_table[src]
            node.name=node.name.replace(src,target)
            for i in range(len(node.input)):
                node.input[i]=node.input[i].replace(src,target)
            for i in range(len(node.output)):
                node.output[i]=node.output[i].replace(src,target)
    return model

# load model
logger.info("step0: load model...")
base_path="/workspace/hgong/onnx_quanter/"
model_name="ResNet_v1_50.onnx"
# model_name="yolov3_darknet_voc.onnx"
# model_name="EfficientNetB2.onnx"
# model_name="ssd_mobilenet_v1_voc.onnx"
# model_name="VGG16.onnx"
# model_name="MobileNetV1.onnx"
onnx_model = onnx.load(base_path+model_name)

strange_symbol_reorientation_table={"@":"_"}
# model shape inference
logger.info("step1: strange_symbol_reorientation...")
onnx_model = replace_strange_symble(onnx_model,strange_symbol_reorientation_table)
# convert model input node to start nodes
logger.info("step2: folding conv and bn...")
optimized_model=conv_bn_fold(onnx_model)
logger.info("step3: finish folding conv and bn")

new_onnx_model_name=base_path+"bn_fold_"+model_name
save_model(optimized_model, new_onnx_model_name)



