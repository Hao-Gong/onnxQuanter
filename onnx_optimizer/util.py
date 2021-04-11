
import onnx
from onnx.tools import update_model_dims
import onnx.helper as helper
from onnx import shape_inference, TensorProto, version_converter
from onnx import optimizer
from onnx import numpy_helper

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