import tensorflow as tf
from google.protobuf import json_format, text_format
from tensorflow.contrib import graph_editor as ge


def save():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, name="input")
        a = tf.Variable(5.0)
        res: tf.Tensor = tf.multiply(a, x, name="mul")

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        g: tf.Graph = tf.get_default_graph()
        gdef = g.as_graph_def()
        gdef = tf.graph_util.convert_variables_to_constants(
            sess,
            gdef, ["mul"],
            variable_names_whitelist=None,
            variable_names_blacklist=None)
        tf.train.write_graph(gdef, logdir="/tmp/k", name="test")


def load():
    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:
            x = tf.placeholder(tf.float32)
            xx = 2 * x + 7
            with open("/tmp/k/test", 'rb') as f:
                graph_def = tf.GraphDef()
                text_format.Merge(f.read(), graph_def)

            y = tf.import_graph_def(
                graph_def,
                input_map={
                    "input:0": xx,
                },
                return_elements=["mul:0"],
                name=None,
                op_dict=None,
                producer_op_list=None)
            print(sess.run(y, feed_dict={
                x: 15,
            }))


def main():
    # save()
    load()
    with tf.Graph().as_default():
        rr = tf.constant(15.0)

        with tf.Session() as sess:
            meta_graph_def = tf.saved_model.loader.load(
                sess, [tf.saved_model.tag_constants.SERVING], "/tmp/lll")

            signature = meta_graph_def.signature_def
            signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            x_tensor_name = signature[signature_key].inputs["x"].name
            y_tensor_name = signature[signature_key].outputs["y"].name

            x = sess.graph.get_tensor_by_name(x_tensor_name)
            y = sess.graph.get_tensor_by_name(y_tensor_name)

            h = tf.get_session_handle(rr)
            h = sess.run(h)

            y_out = sess.run(y, {x: h})
            # print(y_out)

            # for op in tf.get_default_graph().get_operations():
            #     print(op.name)


if __name__ == "__main__":
    main()
