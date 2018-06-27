
from grpc.beta import implementations
import numpy as np

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


host, port = "localhost", 9001
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Send request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'default'
request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


request.inputs['logits'].CopyFrom(
    tf.make_tensor_proto(-np.arange(18, dtype=np.float32).reshape((1, 2, 9)))
)

result = stub.Predict(request, 10.0)  # 10 secs timeout
print(result)
