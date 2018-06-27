# Copyright 2018 Google Inc. All Rights Reserved.
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
# ==============================================================================
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_serving/apis/model_service.proto
# To regenerate run
# python -m grpc.tools.protoc --python_out=. --grpc_python_out=. -I. tensorflow_serving/apis/model_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mincall.external.tensorflow_serving.apis import get_model_status_pb2 as tensorflow__serving_dot_apis_dot_get__model__status__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_serving/apis/model_service.proto',
  package='tensorflow.serving',
  syntax='proto3',
  serialized_pb=_b('\n+tensorflow_serving/apis/model_service.proto\x12\x12tensorflow.serving\x1a.tensorflow_serving/apis/get_model_status.proto2w\n\x0cModelService\x12g\n\x0eGetModelStatus\x12).tensorflow.serving.GetModelStatusRequest\x1a*.tensorflow.serving.GetModelStatusResponseB\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow__serving_dot_apis_dot_get__model__status__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)





DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\370\001\001'))
try:
  # THESE ELEMENTS WILL BE DEPRECATED.
  # Please use the generated *_pb2_grpc.py files instead.
  import grpc
  from grpc.framework.common import cardinality
  from grpc.framework.interfaces.face import utilities as face_utilities
  from grpc.beta import implementations as beta_implementations
  from grpc.beta import interfaces as beta_interfaces


  class ModelServiceStub(object):
    """ModelService provides access to information about model versions
    that have been handled by the model server.
    """

    def __init__(self, channel):
      """Constructor.

      Args:
        channel: A grpc.Channel.
      """
      self.GetModelStatus = channel.unary_unary(
          '/tensorflow.serving.ModelService/GetModelStatus',
          request_serializer=tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusRequest.SerializeToString,
          response_deserializer=tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusResponse.FromString,
          )


  class ModelServiceServicer(object):
    """ModelService provides access to information about model versions
    that have been handled by the model server.
    """

    def GetModelStatus(self, request, context):
      """Gets status of model. If the ModelSpec in the request does not specify
      version, information about all versions of the model will be returned. If
      the ModelSpec in the request does specify a version, the status of only
      that version will be returned.
      """
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')


  def add_ModelServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'GetModelStatus': grpc.unary_unary_rpc_method_handler(
            servicer.GetModelStatus,
            request_deserializer=tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusRequest.FromString,
            response_serializer=tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'tensorflow.serving.ModelService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


  class BetaModelServiceServicer(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    """ModelService provides access to information about model versions
    that have been handled by the model server.
    """
    def GetModelStatus(self, request, context):
      """Gets status of model. If the ModelSpec in the request does not specify
      version, information about all versions of the model will be returned. If
      the ModelSpec in the request does specify a version, the status of only
      that version will be returned.
      """
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


  class BetaModelServiceStub(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    """ModelService provides access to information about model versions
    that have been handled by the model server.
    """
    def GetModelStatus(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      """Gets status of model. If the ModelSpec in the request does not specify
      version, information about all versions of the model will be returned. If
      the ModelSpec in the request does specify a version, the status of only
      that version will be returned.
      """
      raise NotImplementedError()
    GetModelStatus.future = None


  def beta_create_ModelService_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_deserializers = {
      ('tensorflow.serving.ModelService', 'GetModelStatus'): tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusRequest.FromString,
    }
    response_serializers = {
      ('tensorflow.serving.ModelService', 'GetModelStatus'): tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusResponse.SerializeToString,
    }
    method_implementations = {
      ('tensorflow.serving.ModelService', 'GetModelStatus'): face_utilities.unary_unary_inline(servicer.GetModelStatus),
    }
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)


  def beta_create_ModelService_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_serializers = {
      ('tensorflow.serving.ModelService', 'GetModelStatus'): tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusRequest.SerializeToString,
    }
    response_deserializers = {
      ('tensorflow.serving.ModelService', 'GetModelStatus'): tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusResponse.FromString,
    }
    cardinalities = {
      'GetModelStatus': cardinality.Cardinality.UNARY_UNARY,
    }
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'tensorflow.serving.ModelService', cardinalities, options=stub_options)
except ImportError:
  pass
# @@protoc_insertion_point(module_scope)
