from collections import OrderedDict

from typing import List, Dict, Union, Optional, Tuple, Sequence
import copy

import onnx  # type: ignore
import onnx.helper  # type: ignore
import onnx.shape_inference  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnxruntime as rt  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Union, Optional, Tuple, Sequence
import copy

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]

def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
    """
    Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
    """
    v = get_value_info_all(m, name)
    if v is not None:
        return get_shape_from_value_info_proto(v)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))

def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = list(set([ipt.name for ipt in model.graph.input]) -
                       set([x.name for x in model.graph.initializer]))
    return input_names

def get_value_info_all(m: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    for v in m.graph.value_info:
        if v.name == name:
            return v

    for v in m.graph.input:
        if v.name == name:
            return v

    for v in m.graph.output:
        if v.name == name:
            return v

    return None

def generate_rand_input(model, input_shapes: Optional[TensorShapes] = None):
    if input_shapes is None:
        input_shapes = {}
    input_names = get_input_names(model)
    full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
    assert None not in input_shapes
    full_input_shapes.update(input_shapes)  # type: ignore
    for key, shape in full_input_shapes.items():
        if not np.all(np.array(shape) > 0):
            raise RuntimeError(
                'The shape of input "{}" has dynamic size, '
                'please determine the input size manually by --input-shape xxx'.format(key))

    inputs = {ipt: np.array(np.random.rand(*full_input_shapes[ipt]),
                            dtype=get_np_type_from_elem_type(get_elem_type(model, ipt))) for ipt in
              input_names}
    return inputs

def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]


def check_and_update_input_shapes(model: onnx.ModelProto, input_shapes: TensorShapes) -> TensorShapes:
    input_names = get_input_names(model)
    if None in input_shapes:
        if len(input_names) == 1:
            input_shapes[input_names[0]] = input_shapes[None]
            del input_shapes[None]
        else:
            raise RuntimeError(
                'The model has more than 1 inputs, please use the format "input_name:dim0,dim1,...,dimN" in --input-shape')
    for x in input_shapes:
        if x not in input_names:
            raise RuntimeError(
                'The model doesn\'t have input named "{}"'.format(x))
    return input_shapes


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except:
        pass
    return model

def attribute_to_dict(attribute):
    attri_dict={}
    for att in attribute:
        attri_dict[att.name]=onnx.helper.get_attribute_value(att)
    return attri_dict

def infer_model_shape(onnx_model):
    input_tensor = onnx_model.graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape.dim
    input_tensor_new = onnx.helper.make_tensor_value_info(name = input_tensor.name, elem_type = 1, 
                                                        shape = [1, input_shape[1].dim_value, input_shape[2].dim_value, input_shape[3].dim_value])
    onnx_model.graph.input.remove(input_tensor)
    onnx_model.graph.input.insert(0, input_tensor_new)
    tensors = onnx_model.graph.initializer
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = u'1'
    for i, tensor in enumerate(tensors):
        value_info = onnx.helper.make_tensor_value_info(tensor.name, ONNX_DTYPE[tensor.data_type], tensor.dims)
        onnx_model.graph.input.insert(i+1, value_info) # because 0 is for placeholder, so start index is 1
    inferred_onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return inferred_onnx_model

def get_tensor_in_constant(name,nodes):
    for node in nodes:
        if node.op_type=="Constant":
            if node.output==[name]:
                return onnx.numpy_helper.to_array(node.attribute[0].t)
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

def replace_strange_symble(model, strange_symbol_reorientation_table):
    nodes = model.graph.node
    for node in nodes:
        for src in strange_symbol_reorientation_table.keys():
            target=strange_symbol_reorientation_table[src]
            node.name=node.name.replace(src,target)
            for i in range(len(node.input)):
                node.input[i]=node.input[i].replace(src,target)
            for i in range(len(node.output)):
                node.output[i]=node.output[i].replace(src,target)
    return model

def get_node_type_by_input(name,nodes):
    for node in nodes:
        if name in node.input:
            return node.op_type

def get_node_type_by_output(name,nodes):
    for node in nodes:
        if name in node.output:
            return node.op_type

def get_node_by_input(name,nodes):
    for node in nodes:
        if name in node.input:
            return node