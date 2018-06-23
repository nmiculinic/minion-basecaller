import sys
import threading

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


host, port = "localhost", 9000
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Send request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'default'
request.model_spec.signature_name = 'serving_default'

request.inputs['x'].CopyFrom(
    tf.make_tensor_proto(np.array([1, 2, 3, 4, 5], dtype=np.int32))
)

result = stub.Predict(request, 10.0)  # 10 secs timeout
print(result)
