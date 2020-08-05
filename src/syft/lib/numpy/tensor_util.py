# Copyright 2019 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: npuichigo@gmail.com (zhangyuchao)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ...proto.lib.numpy import tensor_pb2


def tensor_to_numpy_array(tensor):
    if tensor.data_type == tensor_pb2.TensorProto.FLOAT:
        return np.asarray(tensor.float_data, dtype=np.float32).reshape(tensor.dims)
    elif tensor.data_type == tensor_pb2.TensorProto.DOUBLE:
        return np.asarray(tensor.double_data, dtype=np.float64).reshape(tensor.dims)
    elif tensor.data_type == tensor_pb2.TensorProto.INT32:
        return np.asarray(tensor.int32_data, dtype=np.int).reshape(
            tensor.dims
        )  # pb.INT32=>np.int use int32_data
    elif tensor.data_type == tensor_pb2.TensorProto.INT16:
        return np.asarray(tensor.int32_data, dtype=np.int16).reshape(
            tensor.dims
        )  # pb.INT16=>np.int16 use int32_data
    elif tensor.data_type == tensor_pb2.TensorProto.UINT16:
        return np.asarray(tensor.int32_data, dtype=np.uint16).reshape(
            tensor.dims
        )  # pb.UINT16=>np.uint16 use int32_data
    elif tensor.data_type == tensor_pb2.TensorProto.INT8:
        return np.asarray(tensor.int32_data, dtype=np.int8).reshape(
            tensor.dims
        )  # pb.INT8=>np.int8 use int32_data
    elif tensor.data_type == tensor_pb2.TensorProto.UINT8:
        return np.asarray(tensor.int32_data, dtype=np.uint8).reshape(
            tensor.dims
        )  # pb.UINT8=>np.uint8 use int32_data
    else:
        # TODO: complete the data type: bool, float16, byte, int64, string
        raise RuntimeError(
            "Tensor data type not supported yet: " + str(tensor.data_type)
        )


def numpy_array_to_tensor(arr):
    tensor = tensor_pb2.TensorProto()
    tensor.dims.extend(arr.shape)
    if arr.dtype == np.float32:
        tensor.data_type = tensor_pb2.TensorProto.FLOAT
        tensor.float_data.extend(list(arr.flatten().astype(float)))
    elif arr.dtype == np.float64:
        tensor.data_type = tensor_pb2.TensorProto.DOUBLE
        tensor.double_data.extend(list(arr.flatten().astype(np.float64)))
    elif arr.dtype == np.int or arr.dtype == np.int32:
        tensor.data_type = tensor_pb2.TensorProto.INT32
        tensor.int32_data.extend(arr.flatten().astype(np.int).tolist())
    elif arr.dtype == np.int16:
        tensor.data_type = tensor_pb2.TensorProto.INT16
        tensor.int32_data.extend(
            list(arr.flatten().astype(np.int16))
        )  # np.int16=>pb.INT16 use int32_data
    elif arr.dtype == np.uint16:
        tensor.data_type = tensor_pb2.TensorProto.UINT16
        tensor.int32_data.extend(
            list(arr.flatten().astype(np.uint16))
        )  # np.uint16=>pb.UNIT16 use int32_data
    elif arr.dtype == np.int8:
        tensor.data_type = tensor_pb2.TensorProto.INT8
        tensor.int32_data.extend(
            list(arr.flatten().astype(np.int8))
        )  # np.int8=>pb.INT8 use int32_data
    elif arr.dtype == np.uint8:
        tensor.data_type = tensor_pb2.TensorProto.UINT8
        tensor.int32_data.extend(
            list(arr.flatten().astype(np.uint8))
        )  # np.uint8=>pb.UNIT8 use int32_data
    else:
        # TODO: complete the data type: bool, float16, byte, int64, string
        raise RuntimeError("Numpy data type not supported yet: " + str(arr.dtype))
    return tensor
