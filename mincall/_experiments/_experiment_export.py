import tensorflow as tf
from mincall.basecall.strategies import BeamSearchSess
import os

with tf.Session() as sess:
    bs = BeamSearchSess(sess=sess, surrogate_base_pair=True)

    model_input = tf.saved_model.utils.build_tensor_info(bs.logits_ph)
    model_output = tf.saved_model.utils.build_tensor_info(bs.predict_values)

    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"logits": model_input},
        outputs={"path": model_output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    export_path = os.path.join("/tmp", "bs", "1")
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition,
        },
    )
    builder.save()
