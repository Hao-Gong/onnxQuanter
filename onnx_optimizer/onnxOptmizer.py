
# onnx-simplifier/onnxsim/onnx_simplifier.py /
# @daquexian

from collections import OrderedDict

from typing import List, Dict, Union, Optional, Tuple, Sequence
import copy

import onnx  # type: ignore
import onnx.helper  # type: ignore
import onnx.shape_inference  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnxruntime as rt  # type: ignore
# import onnxoptimizer  # type: ignore

import numpy as np  # type: ignore
from typing import List, Dict, Union, Optional, Tuple, Sequence
import copy

from .onnxParser import onnxParser
from .onnxRTEngine import onnxRTEngine

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]

class onnxOptEngine(onnxRTEngine):
    def __init__(self,cfg:dict):
        self.cfg=cfg
